#Siamese Neural Network
import tensorflow as tf
from Train_05_03_22 import *
#Keras
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.models import load_model
#Avoiding GPU Memory error
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

#Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
#Make predictions
y_hat = siamese_model.predict([test_input, test_val])
print(y_hat)
#Post-process results
res = []
for prediction in y_hat:
    if prediction > 0.500:
        res.append(1)
    else:
        res.append(0)
print(res)
print(y_true)
r = Precision()
r.update_state(y_true, y_hat)
print(r.result().numpy())
#if __name__ == '__main__':
    #Save Model
    #siamese_model.save('D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/SIAM/siamesemodel.h5')
    #Reload model
    #model = load_model('D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/SIAM/siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    #print(model.predict([test_input, test_val]))

#test_var = test_input, test_val, y_true = test_data.as_numpy_iterator().next()
#print(test_var[2])

