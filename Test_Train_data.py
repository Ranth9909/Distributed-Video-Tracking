import tensorflow as tf
from Collect_Train_data import *
from Train_18_01_22 import *
import imutils

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('PATH/application_data', 'verification_images')):
         input_img = preprocess(os.path.join('PATH/application_data', 'input_image', 'input_image.png'))
         validation_img = preprocess(os.path.join('PATH/application_data', 'verification_images', image))
         result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
         #print(model.predict([test_input, test_val]))
         results.append(result)
    detection = np.sum(np.array(results) > detection_threshold)
    #print(detection)
    verification = detection / len(os.listdir(os.path.join('D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/SIAM3/application_data', 'verification_images')))
    verified = verification > verification_threshold
    return results, verified

cap = cv2.VideoCapture('VIDEO_INPUT') #any video input
if __name__ == "__main__":
    model_path = 'PATH/SORT/faster_rcnn_inception_v2/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.5
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        boxes, scores, classes, num = odapi.processFrame(frame)
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                objectID = "ID {}".format(i)         
                cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(208,224,64),1)
                cropped_image = frame[box[0]+2:box[2]-2,box[1]+2:box[3]-2]
                path = 'PATH/application_data/input_image/input_image.png'
                cv2.imwrite(path,cropped_image)
                image = Image.open(path)
                new_image = image.resize((128, 128))
                new_image.save('PATH/application_data/input_image/input_image.png')
        cv2.imshow('Verification', frame)
        #Verification trigger
        if cv2.waitKey(1) & 0xFF == ord('a'):
            #Save image input
            #cv2.imwrite('D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/SIAM/application_data/input_image/input_image.png',frame)
            model1 = load_model('PATH/siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
            results, verified = verify(model1,0.7,0.5)
            print(verified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

