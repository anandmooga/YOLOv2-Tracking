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
import math
import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vpath', action = 'store', dest = 'video_path', help = 'Read frame path')
parser.add_argument('-l', action = 'store', dest = 'layer', help = 'Layer in int from where we get embedding', type = int)

results = parser.parse_args()
VIDEO_DIR = results.video_path 
layer = results.layer

if VIDEO_DIR == None:
	VIDEO_DIR = "Video_tsne/"

if layer == None:
	layer = 13

#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolov2_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolov2.h5")

#Print the summery of the model
yolo_model.summary()

#13, 14, 15 startiing from 1
yolo_model_cut = Sequential()
for i in range(layer):
	yolo_model_cut.add(yolo_model.layers[i])

yolo_model_cut.compile(loss = "categorical_crossentropy", optimizer = SGD())

yolo_model_cut.summary()


video = os.listdir(VIDEO_DIR)
video.sort()
embeddings = []
colors = generate_colors(class_names)

# Initiate a session
sess = K.get_session()

names = []

for frame in video:
	input_image_name = frame
	#Obtaining the dimensions of the input image
	input_image = Image.open(VIDEO_DIR + input_image_name)
	width, height = input_image.size
	width = np.array(width, dtype=float)
	height = np.array(height, dtype=float)
	image_shape = (height, width)
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
	#Convert final layer features to bounding box parameters
	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
	# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
	boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)
	#Preprocess the input image before feeding into the convolutional network
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
	#Run the session
	out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
	draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	image.save(os.path.join("out", input_image_name), quality=90)
	activation_map = yolo_model_cut.predict(image_data)
	activation_shape = activation_map.shape  #N,W,H,C
	out_boxes[:, [0,2]] *= (activation_shape[1]/image_shape[0])
	out_boxes[:, [1,3]] *= (activation_shape[2]/image_shape[1])
	out_boxes = out_boxes.astype(int)
	z = 1
	for roi in out_boxes:
		flat_mat = []
		for channel in range(activation_shape[3]):
			flat_mat.append(np.reshape(activation_map[0, roi[0]-1:roi[2]+1, roi[1]-1:roi[3]+1, channel], -1))
		flat_mat = np.array(flat_mat)
		flat_mat_t = np.transpose(flat_mat)
		#print(np.array(flat_mat).shape, np.array(flat_mat_t).shape)
		gram_mat = np.matmul(flat_mat, flat_mat_t)
		#print(np.reshape(gram_mat, -1).shape)
		embeddings.append(np.reshape(gram_mat, -1))
		names.append("f" + frame[5:10]+ "_" + class_names[out_classes[z-1]] + "_" +  str(z))
		z += 1
	print(frame)

	#if frame == "frame_0003.jpg":
	#	break




import pandas as pd

dfe = pd.DataFrame(embeddings)
print(dfe.head())
dfe.to_csv("embeddings.tsv",sep = "\t", header = False, index = False)

dfn = pd.DataFrame(names)
print(dfn.head())
dfn.to_csv("objects.tsv",sep = "\t", header = False, index = False)


print(len(embeddings))
print(len(names))
