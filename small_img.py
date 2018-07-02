###########################################################################################################################
############################################## MAPPING BACK ISSUE #########################################################
###########################################################################################################################

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
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial import distance

#rectangle packing package 
import rpack
#pip install rectangle-packer
from collections import namedtuple
R = namedtuple('R', 'width height x y')
def enclosing_size(sizes, positions):
    """Return enclosing size of rectangles having sizes and positions"""
    rectangles = [R(*size, *pos) for size, pos in zip(sizes, positions)]
    width = max(r.width + r.x for r in rectangles)
    height = max(r.height + r.y for r in rectangles)
    return width, height

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
anchors = read_anchors("model_data/yolov2_fc_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolov2_fc.h5")

#Print the summery of the model
#yolo_model.summary()

#13, 14, 15 startiing from 1
yolo_model_cut = Sequential()
for i in range(layer):
	yolo_model_cut.add(yolo_model.layers[i])

yolo_model_cut.compile(loss = "categorical_crossentropy", optimizer = SGD())

#yolo_model_cut.summary()


video = os.listdir(VIDEO_DIR)
video.sort()
embeddings = []
extra_box = 0.2
extra_embd = 0.04
pad = 4
colors = generate_colors(class_names)

# Initiate a session
sess = K.get_session()

names = []

no = 0
for frame in video:
	if no == 0:
		input_image_name = frame
		#Obtaining the dimensions of the input image
		input_image = Image.open(VIDEO_DIR + input_image_name)
		width, height = input_image.size
		width = np.array(width, dtype=float)
		height = np.array(height, dtype=float)
		image_shape = (height, width)
		image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
		
		yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
		boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)
		

		out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
		print("Initial bounding box: \n", out_boxes)
		print("Initial bounding box shapes: \n", out_boxes[:, 2:4]-out_boxes[:, 0:2])
		draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
		image.save(os.path.join("out", input_image_name), quality=90)
		
		out_boxes_cut = np.copy(out_boxes)
		out_boxes_cut = out_boxes_cut.astype(int)
		
		activation_map = yolo_model_cut.predict(image_data)
		activation_shape = activation_map.shape  #N,W,H,C
		
		out_boxes[:, [0,2]] *= (activation_shape[1]/image_shape[0])
		out_boxes[:, [1,3]] *= (activation_shape[2]/image_shape[1])
		out_boxes = out_boxes.astype(int)
		

		z = 1
		embeddings_prev = []
		for roi in out_boxes:
			object_list = []
			for channel in range(activation_shape[3]):
				y_per1 = y_per2 = int((roi[2]-roi[0])*extra_embd)
				x_per1 = x_per2 = int((roi[3]-roi[1])*extra_embd)
				if(roi[0] - y_per1 < 0):
					y_per1 = 0
				if(roi[2] + y_per2 > image_shape[0]):
					y_per2 = 0
				if(roi[1] - x_per1 < 0):
					x_per1 = 0
				if(roi[3] + x_per2 > image_shape[1]):
					x_per2 = 0
				object_list.append(np.average(activation_map[0, roi[0]-y_per1:roi[2]+y_per2, roi[1]-x_per1:roi[3]+x_per2, channel]))
			embeddings.append(object_list)
			embeddings_prev.append(object_list)
			names.append("f" + frame[5:10]+ "_" + class_names[out_classes[z-1]] + "_" +  str(z))
			z += 1
		
		print(frame)
		no += 1

	else:
		input_image_name = frame
		#Obtaining the dimensions of the input image
		input_image = Image.open(VIDEO_DIR + input_image_name)
		width, height = input_image.size
		width = np.array(width, dtype=float)
		height = np.array(height, dtype=float)
		image_shape = (height, width)
		image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
		image_full = np.array(image)

		cut_sizes = []
		cuts = []
		#incorportae padding !!!!!
		print("Cut bounding box{}: \n".format(no), out_boxes_cut)
		print("Initial bounding box{} shapes: \n".format(no), out_boxes_cut[:, 2:4]-out_boxes_cut[:, 0:2])
		cnt = 0
		out_boxes_cut_edges = np.zeros((out_boxes_cut.shape[0], 2))
		for roi in out_boxes_cut:
			y_per1 = y_per2 = int((roi[2]-roi[0])*extra_box)
			x_per1 = x_per2 = int((roi[3]-roi[1])*extra_box)
			if(roi[0] - y_per1 < 0):
				y_per1 = 0
			if(roi[2] + y_per2 > image_shape[0]):
				y_per2 = 0
			if(roi[1] - x_per1 < 0):
				x_per1 = 0
			if(roi[3] + x_per2 > image_shape[1]):
				x_per2 = 0
			print(y_per1, y_per2, x_per1, x_per2)
			cut_out = np.pad(image_full[roi[0]-y_per1:roi[2]+y_per2, roi[1]-x_per1:roi[3]+x_per2, :], [(pad,pad), (pad,pad), (0,0)], "constant")
			cuts.append(cut_out)
			cut_sizes.append(list(cut_out.shape[0:2]))
			out_boxes_cut_edges[cnt] = [roi[0]-y_per1 - pad, roi[1]-x_per1 -pad]
			cnt += 1
			#print(cut_out.shape[0:2])
		cuts = np.array(cuts)
		cut_sizes = np.array(cut_sizes)
		print("Cut bounding box{} shapes: \n".format(no), cut_sizes)
		ind = np.argsort(cut_sizes[:,0]) #sorting by height
		print("corresponding indices at {} frame \n".format(no), ind[::-1])
		cuts = cuts[ind[::-1]]
		cut_sizes = cut_sizes[ind[::-1]]
		cut_sizes = cut_sizes.tolist()
		positions = rpack.pack(cut_sizes)
		print("reshaped_image using postitons: \n", positions)
		out_boxes_cut = out_boxes_cut[ind[::-1]]
		out_boxes_cut_edges = out_boxes_cut_edges[ind[::-1]]
		print("Initial bounding box{} shapes after sorting: \n".format(no), out_boxes_cut[:, 2:4]-out_boxes_cut[:, 0:2])
		#print(positions)
		w, h = enclosing_size(cut_sizes, positions)
		#print(w,h)

		#creating new image ! 
		if (w > h):
			if w % 32 != 0:
				wf = w +  32 - (w%32)
				reshaped_image = np.zeros((wf, wf, 3)).astype(int)
				wf = np.array(wf, dtype=float)
				image_shape_reshaped = (wf, wf)
			else:
				hf = h +  32 - (h%32)
				reshaped_image = np.zeros((hf, hf, 3)).astype(int)
				hf = np.array(hf, dtype=float)
				image_shape_reshaped = (hf, hf)

		for i in range(len(positions)):
			reshaped_image[positions[i][0]:positions[i][0]+cut_sizes[i][0], positions[i][1]:positions[i][1]+cut_sizes[i][1], :] = cuts[i]
		reshaped_image = reshaped_image/255
		############cut image made ###########


		image_data = reshaped_image[np.newaxis, :, : , :]
		image_pil_reshaped = Image.fromarray(np.uint8(reshaped_image*255))

		#Convert final layer features to bounding box parameters
		yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
		boxes, scores, classes = yolo_eval(yolo_outputs, image_shape_reshaped, score_threshold = 0.45)

		#Run the session
		out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
		draw_boxes(image_pil_reshaped, out_scores, out_boxes, out_classes, class_names, colors)
		image_pil_reshaped.save(os.path.join("out", input_image_name), quality=90)
		 
		out_boxes_reshaped = np.copy(out_boxes)
		out_boxes_reshaped = out_boxes_reshaped.astype(int)
		#get the embeddings
		activation_map = yolo_model_cut.predict(image_data)
		activation_shape = activation_map.shape  #N,W,H,C
		out_boxes[:, [0,2]] *= (activation_shape[1]/image_shape[0])
		out_boxes[:, [1,3]] *= (activation_shape[2]/image_shape[1])
		out_boxes = out_boxes.astype(int)


		z = 1
		embeddings_new = []
		for roi in out_boxes:
			object_list = []
			for channel in range(activation_shape[3]):
				y_per1 = y_per2 = int((roi[2]-roi[0])*extra_embd)
				x_per1 = x_per2 = int((roi[3]-roi[1])*extra_embd)
				if(roi[0] - y_per1 < 0):
					y_per1 = 0
				if(roi[2] + y_per2 > image_shape[0]):
					y_per2 = 0
				if(roi[1] - x_per1 < 0):
					x_per1 = 0
				if(roi[3] + x_per2 > image_shape[1]):
					x_per2 = 0
				object_list.append(np.average(activation_map[0, roi[0]-y_per1:roi[2]+y_per2, roi[1]-x_per1:roi[3]+x_per2, channel]))
			embeddings.append(object_list)
			embeddings_new.append(object_list)
			names.append("f" + frame[5:10]+ "_" + class_names[out_classes[z-1]] + "_" +  str(z))
			z += 1

		#finding the corresponding prev bounding box !
		#X = embeddings_new + embeddings_prev
		#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X) #2 becasse one to one mapping ! 
		#distances, indices = nbrs.kneighbors(X)

		print("Cut bounding box in embeded space{}: \n".format(no), out_boxes_reshaped)
		print("Initial bounding box{} shapes in embeded space: \n".format(no), out_boxes_reshaped[:, 2:4]-out_boxes_reshaped[:, 0:2])
		embeded_box_shape = out_boxes_reshaped[:, 2:4]-out_boxes_reshaped[:, 0:2]
		distances = distance.cdist(np.array(positions), out_boxes_reshaped[:, 0:2])
		print("HREE IDSDSD")
		print(positions)
		print(out_boxes_reshaped[:, 0:2])
		print(np.argmin(distances, axis = 1))

		mapping = np.argmin(distances, axis = 1)
		out_boxes_offset = out_boxes_reshaped[mapping]
		out_boxes_offset[:, 0:2] = out_boxes_offset[:, 0:2]-np.array(positions) 
		out_boxes_offset[:, 2:4] = out_boxes_offset[:, 2:4]-np.array(positions) 
		offsets = out_boxes_offset[:, 0:2]-np.array(positions) 
		offsets = out
		#for i in out_boxes_cut:
		#print(out_boxes_reshaped[:, 0:2])
		print("Bounding box offsets to be mapped back for {} run: \n".format(no), out_boxes_offset)

		cnt =0
		out_boxes_cut_new = np.zeros(out_boxes_reshaped.shape)
		for i in mapping:
			roi = out_boxes_cut_edges[cnt]
			print(y_per1, y_per2, x_per1, x_per2)
			out_boxes_cut_new[cnt] = np.array([ roi[0]+out_boxes_offset[i][0] , roi[1]+out_boxes_offset[i][1],  roi[0]+out_boxes_offset[i][0]+embeded_box_shape[i][0], roi[1]+out_boxes_offset[i][1]+embeded_box_shape[i][1]])
			cnt += 1
			if cnt> out_boxes_reshaped.shape[0]:
				break
		out_boxes_cut = out_boxes_cut_new.astype(int)
		draw_boxes(image, out_scores, out_boxes_cut, out_classes, class_names, colors)
		plt.imshow(image)
		plt.show()
		break
		print("Mapped back bounding boxes for {} run \n".format(no), out_boxes_cut)
		print("Mapped bounding box{} shapes: \n".format(no), out_boxes_cut[:, 2:4]-out_boxes_cut[:, 0:2])
		print(frame)
		no += 1
		#break

import pandas as pd

dfe = pd.DataFrame(embeddings)
#print(dfe.head())
dfe.to_csv("embeddings.tsv",sep = "\t", header = False, index = False)

dfn = pd.DataFrame(names)
#print(dfn.head())
dfn.to_csv("objects.tsv",sep = "\t", header = False, index = False)


print(len(embeddings))
print(len(names))




		


































