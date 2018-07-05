# import the needed modules
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval


#Provide the name of the image that you saved in the images folder to be fed through the network
input_image_name = "vidhi1.jpg"

#Obtaining the dimensions of the input image
input_image = Image.open("images/" + input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
print(class_names)
anchors = read_anchors("model_data/yolov2_fc_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolov2_fc.h5")

#Print the summery of the model
yolo_model.summary()




#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)


# Initiate a session
sess = K.get_session()

input_size = 32*13

'''MAKE SURE THAT INPUT IS SQUARE, FOR FULLY CONVOLUTIONAL NETWORK, ALSO 32 IS YOLOV2'S DOWNSAMPLING'''
#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image("images/" + input_image_name, model_image_size = (input_size, input_size))
maps = yolo_model.predict(image_data)
print(maps.shape)

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
ppl_ind = np.argwhere(out_classes == 0)
ppl_ind = np.squeeze(ppl_ind)
print(out_classes, ppl_ind)
out_scores = out_scores[ppl_ind]
out_boxes = out_boxes[ppl_ind]
out_classes = out_classes[ppl_ind]

print(out_boxes, type(out_boxes))
print(out_scores)
print(out_classes)
#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out", input_image_name))
imshow(output_image)
plt.show()