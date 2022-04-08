from inspect import currentframe
import numpy as np
import tensorflow as tf
import cv2
import imutils
import os
import shutil
from PIL import Image

currentFrame = 0
j = 0
cur_strg = ""
path1= 'D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/frames/'
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":
    model_path = 'D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/FYP2/AIComputerVision-master/SORT/faster_rcnn_inception_v2/frozen_inference_graph.pb' 
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.5
    #cap = cv2.VideoCapture('http://admin:OBtHmuSP@192.168.0.30/VIDEO.CGI')
    #cap = cv2.VideoCapture('test_video.mp4')
    cap = cv2.VideoCapture('IMG_3585.MOV')
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    print("Height:", height)
    print("Width", width)


    while True:
        r, img = cap.read()
        img = imutils.resize(img, width=600)
        #img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                objectID = "ID {}".format(i)         
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(208,224,64),1)
                cv2.putText(img,objectID,(box[1],box[0]-5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (208,224,64),1)
                cropped_image = img[box[0]+2:box[2]-2,box[1]+2:box[3]-2]
                path = 'D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/frames/ID'+str(i)+'_FRAME_' + cur_strg+'.png'
                cv2.imwrite(path,cropped_image)
                image = Image.open(path)
                new_image = image.resize((128, 128))
                new_image.save('D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/frames/ID'+str(j)+'_FRAME_' + cur_strg+'.png')
                currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cur_strg = str(currentFrame)
                cur_strg = cur_strg.replace(".0","")

        cv2.imshow("IMG", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            dir_list = os.listdir(path1)
            for f in dir_list:
                dir_name = f[:4]
                print(dir_name)
                dir_path = path1 + dir_name
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                if os.path.exists(dir_path):
                    file_path = path1 + f
                    shutil.move(file_path, dir_path)
            break



    
