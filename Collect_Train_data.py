#Siamese Neural Network
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
#Keras
from time import time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
#from torch import embedding
#Avoiding GPU Memory error
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Paths
POS_PATH = os.path.join('PATH/data', 'positive')
NEG_PATH = os.path.join('PATH/data', 'negative')
ANC_PATH = os.path.join('PATH/data', 'anchor')
#Directory
#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

#Get Image Directory
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.png').take(200)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.png').take(200)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.png').take(200)

#Preprocessing
def preprocess(file_path): 
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img / 255.0
    return img

#Create Labelled Dataset- 
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

#Twin isntead of triple loss
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

#Building Dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
samples = data.as_numpy_iterator()
samp = samples.next()

#Train Partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(32)
train_data = train_data.prefetch(16)
#print(round(len(data)*.7))
#Test Partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(32)
test_data = test_data.prefetch(16)

#Build Embedding Layer
def make_embedding():
    inp = Input(shape=(105,105,3), name = 'input_image')
    #First Block
    c1 = Conv2D(64,(10,10), activation = 'relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding = 'same')(c1)
    drop1 = Dropout(0.25)(m1)
    #Second Block
    c2 = Conv2D(128,(7,7), activation = 'relu')(drop1)
    m2 = MaxPooling2D(64, (2,2), padding = 'same')(c2)
    #Third Block
    c3 = Conv2D(128,(4,4), activation = 'relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding = 'same')(c3)
    #Final Block
    c4 = Conv2D(256,(4,4), activation = 'relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=[inp], outputs=[d1], name='embedding')
embedding = make_embedding()
#embedding.summary()

#Build Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    #Similiarity Check
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

#Siamese Model
def make_siamese_model():
    #Anchor Image
    input_image = Input(name='input_img', shape=(105,105,3))
    #Validation Image
    validation_image = Input(name='validation_img', shape=(105,105,3))
    #Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    #Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
siamese_model = make_siamese_model()
#siamese_model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Setup Loss & Optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(5e-4) #0.0005 
#Checkpoints
checkpoint_dir = 'PATH/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
#Build Train Step Function
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    print(loss)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss
#Build Training Loop
def train(data, EPOCHS):
    #loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        r = Recall()
        p = Precision()
        #loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())  
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == '__main__':
    EPOCHS = 10
    train(train_data, EPOCHS)
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    #Make predictions
    y_hat = siamese_model.predict([test_input, test_val])
    #print(y_hat)
    #Post-process results
    res = []
    for prediction in y_hat:
        if prediction > 0.5:
            res.append(1)
        else:
            res.append(0)
    print(res)
    print(y_true)
    r = Precision()
    r.update_state(y_true, y_hat)
    print(r.result().numpy())
    #Save Model
    siamese_model.save('PATH/siamesemodel.h5')
    #Reload model
    model = load_model('PATH/siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    #print(model.predict([test_input, test_val]))

