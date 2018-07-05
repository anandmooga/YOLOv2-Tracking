# import the needed modules
import os
import cv2
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


def box_per(out_box, hpercent = 0.2, wpercent = 2*0.15):
	'''
	returns (y,x) to be added or subtracted !
	'''
	h = out_box[2] - out_box[0]
	w = out_box[3] - out_box[1]

	return(int(h*hpercent), int(w*wpercent))

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
#extra_box = 0.2
extra_embd = 0.04
pad = 4
colors = generate_colors(class_names)
to_plot = False


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

		## Saving previous bounding boxes
		out_boxes_cut = np.copy(out_boxes)
		out_boxes_cut = out_boxes_cut.astype(int)

		'''
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
		'''
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

		#Cutting the new image on basis of prev bb
		cut_images = []
		cut_images_sizes = []
		cut_edges = []
		print("BOXES WE ARE GOING TO CUT WITH ! :")
		for roi in out_boxes_cut:
			#getting the extra margin 
			dely, delx = box_per(roi)
			dely1 = dely2 = dely
			delx1 = delx2 = delx
			#cutting image from new image on basis of previous bb
			if(roi[0] - dely < 0):
				dely1 = 0
			if(roi[2] + dely > image_shape[0]):
				dely2 = 0
			if(roi[1] - delx < 0):
				delx1 = 0
			if(roi[3] + delx > image_shape[1]):
				delx2 = 0
			cut_out = image_full[roi[0]-dely1:roi[2]+dely2, roi[1]-delx1:roi[3]+delx2, :]
			cut_edges.append([roi[0]-dely1, roi[1]-delx1])
			print("{}:{} and {}:{}".format(roi[0]-dely1, roi[2]+dely2, roi[1]-delx1, roi[3]+delx2))
			cut_out = np.pad(cut_out, [(pad,pad), (pad,pad), (0,0)], "constant")
			#plt.imshow(cut_out)
			#plt.show()
			cut_images.append(cut_out)
			cut_images_sizes.append(cut_out.shape[0:2])

		cut_images = np.array(cut_images)
		cut_images_sizes = np.array(cut_images_sizes)
		cut_edges = np.array(cut_edges)
		print("Cut sizes: ", cut_images_sizes)

		#now to find the coordinates of the cut outs in the small image
		ind = np.argsort(cut_images_sizes[:,0]) #sorting by height
		ind = ind[::-1]
		#sorting cut_out according to height and also sorting prev_bb accronnding to the same order 
		cut_images = cut_images[ind]
		cut_images_sizes = cut_images_sizes[ind]
		out_boxes_cut = out_boxes_cut[ind]
		cut_edges = cut_edges[ind]


		cut_images_sizes = cut_images_sizes.tolist()

		#genreating co-ordinates
		positions = rpack.pack(cut_images_sizes)
		print("Positins, ", positions)

		#finding out shape of new image
		w, h = enclosing_size(cut_images_sizes, positions)
		#calculating shape of new image! 
		if (w > h):
			if w % 32 != 0:
				wf = w +  32 - (w%32)
				reshaped_image = np.zeros((wf, wf, 3)).astype(int)
				wf = np.array(wf, dtype=float)
				image_shape_reshaped = (wf, wf)
			else:
				reshaped_image = np.zeros((w, w, 3)).astype(int)
				w = np.array(h, dtype=float)
				image_shape_reshaped = (w, w)
		else:
			if h % 32 != 0:
				hf = h +  32 - (h%32)
				reshaped_image = np.zeros((hf, hf, 3)).astype(int)
				hf = np.array(hf, dtype=float)
				image_shape_reshaped = (hf, hf)
			else:
				reshaped_image = np.zeros((h, h, 3)).astype(int)
				h = np.array(h, dtype=float)
				image_shape_reshaped = (h,h)

		position_boxes = []
		for i in range(len(positions)):
			reshaped_image[positions[i][0]:positions[i][0]+cut_images_sizes[i][0], positions[i][1]:positions[i][1]+cut_images_sizes[i][1], :] = cut_images[i]
			position_boxes.append([positions[i][0], positions[i][1], positions[i][0]+cut_images_sizes[i][0], positions[i][1]+cut_images_sizes[i][1]])
		reshaped_image = reshaped_image/255
		if to_plot:
			plt.imshow(reshaped_image)
			plt.show()
		#break
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
		if to_plot:
			plt.imshow(image_pil_reshaped)
			plt.show()
		 
		out_boxes_reshaped = np.copy(out_boxes)
		out_boxes_reshaped = out_boxes_reshaped.astype(int)
		print("new boxes: ", out_boxes_reshaped)

		'''
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
		'''

		#finding the corresponding prev bounding box !
		## To do this we will map distance of current bounding boxes with the cut out box co-ordinates !
		distances = distance.cdist(out_boxes_reshaped, np.array(position_boxes))
		#print(distances)
		mapping = np.argmin(distances, axis = 1)
		#print(mapping)
		conflict_mapping = np.argmin(distances, axis = 0)
		#print(conflict_mapping)
		out_boxes_mapped = []
		good_ind = []
		for ind_curr, ind_prev in enumerate(mapping):
			if conflict_mapping[ind_prev] == ind_curr: #this is the correct mapping
				# ind_prev is the index for the cooresponding prevbb or cut_out where we have to map it !
				#getting prev cut coordinates
				roi =  cut_edges[ind_prev]
				h = out_boxes_reshaped[ind_curr][2] - out_boxes_reshaped[ind_curr][0]
				w = out_boxes_reshaped[ind_curr][3] - out_boxes_reshaped[ind_curr][1]
				offsets = out_boxes_reshaped[ind_curr][0:2] - positions[ind_prev] - np.array([pad, pad])
				print(roi[0], roi[1])
				print(offsets, h, w)
				mapped_back_box = np.array([roi[0]+int(offsets[0]), roi[1]+int(offsets[1]), roi[0]+int(offsets[0]+h), roi[1]+int(offsets[1]+w)])
				out_boxes_mapped.append(mapped_back_box)
				good_ind.append(ind_curr)
			else:
				#wrong mapping, further logic can be added to overcome failure of yolo
				#if something left that was mapped last time i want to keep it in the image!
				


				pass
		good_ind = np.array(good_ind)
		out_boxes_mapped = np.array(out_boxes_mapped)
		out_boxes_cut = out_boxes_mapped
		draw_boxes(image, out_scores[good_ind], out_boxes_cut, out_classes[good_ind], class_names, colors)
		image.save(os.path.join("out_track", input_image_name), quality=90)
		if to_plot:
			plt.imshow(image)
			plt.show()
		print(frame)
		#break
		no += 1

		


	
import pandas as pd

dfe = pd.DataFrame(embeddings)
#print(dfe.head())
#dfe.to_csv("embeddings.tsv",sep = "\t", header = False, index = False)

dfn = pd.DataFrame(names)
#print(dfn.head())
#dfn.to_csv("objects.tsv",sep = "\t", header = False, index = False)


print(len(embeddings))
print(len(names))


