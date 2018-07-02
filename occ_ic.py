
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
#import scipy.io
#import scipy.misc
#from scipy.spatial import distance
import numpy as np
from PIL import Image
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.manifold import TSNE
import cv2

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.models import load_model

from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, draw_boxes_track
from yad2k.models.keras_yolo import yolo_head, yolo_eval
#from sklearn.decomposition import PCA

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

class_names = read_classes("model_data/coco_classes.txt")
#print(class_names)
anchors = read_anchors("model_data/yolov2_fc_anchors.txt")
yolo_model = load_model("model_data/yolov2_fc.h5")
colors = generate_colors(class_names)
track_colors = generate_colors(["object_"+str(i) for i in range(50)])
#yolo_model.summary()

yolo_model_cut = Sequential()
for i in range(layer):
	yolo_model_cut.add(yolo_model.layers[i])
yolo_model_cut.compile(loss = "categorical_crossentropy", optimizer = SGD())
#yolo_model_cut.summary()

video = os.listdir(VIDEO_DIR)
video.sort()
embeddings = []
names = []

object_sv = {} #key is the identifier and value is list of some bb and embedding, has only non faulty objects !
object_count = 0


class object_v():
	'''
	Object that represents an object to be tracked
	'''
	def __init__(self, bounding_box):
		self.bounding_box = bounding_box
		self.last_seen_bb = bounding_box
		self.f_count = 0
		self.occ_flag = False

		self.x0 = bounding_box
		self.prev_boxes = np.empty((0,4))
		self.kernels = []
		self.sort_order = np.array([0,1,2,3,4])
		#x1, x2, x3, x4, x5

	def update(self, bounding_box, kernel, track_interval = 5):
		self.f_count += 1
		if self.occ_flag == False:
			if (self.f_count > track_interval):
				self.sort_order = np.argsort(self.prev_boxes[:, 1], axis = 0)[::-1] #sorting on basis of x-coordinate
				self.prev_boxes[self.sort_order[-1]] = bounding_box
				self.kernels[self.sort_order[-1]] = kernel
				self.sort_order = np.argsort(self.prev_boxes[:, 1], axis = 0)[::-1]
			else:
				self.prev_boxes = np.append(self.prev_boxes, np.array([bounding_box]), axis =0)
				self.kernels.append(kernel)
			self.last_seen_bb = bounding_box

		self.bounding_box = bounding_box #will use to get area to scan
		
		#if self.flag == 0:
		#	self.bounding_box_kernel = self.bounding_box_prev
		#	self.flag = 1
	def flag_set(self, flag):
		self.occ_flag = flag




def initialize_tracker(out_boxes):
	'''
	It creates seprate objects for each object identified in the image. called only once in the begnning
	'''
	global object_sv
	global object_count
	for i in range(len(out_boxes)):
		object_sv["object_" + str(object_count)] = object_v(out_boxes[i])
		object_count += 1

def box_per(out_box, hpercent = 0.1, wpercent = 0.1):
	'''
	returns (y,x) to be added or subtracted !
	'''
	h = out_box[2] - out_box[0]
	w = out_box[3] - out_box[1]
	return(h*hpercent, w*wpercent)

def overlap_checker(bb_1, bb_2, margin = 0.05):
	'''
	[0,1,2,3] are the values of bb, the the bounding box is (0,1)------------(2,1)
															  |                |
														  	  |                |
								  	  						(0,3)------------(2,3)
	'''
	dely , delx = box_per(bb_2, margin, 2*margin)

	if bb_1[0] > bb_2[2]+dely or bb_1[2] < bb_2[0]-dely:
		return False
	if bb_1[1] > bb_2[3]+delx or bb_1[3] < bb_2[1]-delx:
		return False
	return True

def occulsion_detector(margin = 0.05, loc = False):
	occulsion_keys = []
	flag = False
	keys = list(object_sv.keys())
	for i in range(len(keys)):
		for j in range(i+1, len(keys)):
			if overlap_checker(object_sv[keys[i]].bounding_box, object_sv[keys[j]].bounding_box, margin):
				flag = True
				occulsion_keys.append(keys[i])
				occulsion_keys.append(keys[j])

	if loc == False:
		return flag
	else:
		normal_keys = set(keys) - set(occulsion_keys)
		return (flag, set(occulsion_keys), normal_keys)


def kernel_cutter(prev_img, prev_box):
	prev_img_cp = np.array(prev_img)
	prev_img_cp = cv2.cvtColor(prev_img_cp,cv2.COLOR_BGR2GRAY)
	kernel = prev_img_cp[int(prev_box[0]):int(prev_box[2]), int(prev_box[1]):int(prev_box[3])]
	return kernel 


def insane_conv(curr_img, prev_box, kernel, margin = 0.15, matplot = False):
	#prev_box is to cut a big patch from curr_img
	#
	#first we will take a crop out of the current frame on basis of previous bb
	#then we will use insane conv using the prev bounding box precise filter! and get new bb !
	#insance conv kicks in when yolo fails !

	curr_img_cp = np.array(curr_img)
	curr_img_cp = cv2.cvtColor(curr_img_cp,cv2.COLOR_BGR2GRAY)
	#plt.imshow(curr_img_cp)
	#plt.show()
	#print("insance")
	dely, delx = box_per(prev_box, margin, 2*margin)
	#print(int(prev_box[0]-dely), int(prev_box[2]+dely), int(prev_box[1]-delx), int(prev_box[3]+dely))
	image_crop = curr_img_cp[int(prev_box[0]-dely):int(prev_box[2]+dely), int(prev_box[1]-delx):int(prev_box[3]+dely)]
	#print(image_crop.shape)
	w, h = kernel.shape[::-1]
	method = eval('cv2.TM_CCOEFF')
	res = cv2.matchTemplate(image_crop,kernel,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	box = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
	#print(box)
	
	if matplot:
		plt.subplot(121), plt.imshow(kernel, cmap = 'gray')
		plt.subplot(122),plt.imshow(image_crop, cmap = 'gray')
		plt.show()

		cv2.rectangle(image_crop,top_left, bottom_right, 255, 5)
		plt.subplot(121),plt.imshow(res,cmap = 'gray')
		plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(image_crop,cmap = 'gray')
		plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
		plt.suptitle('cv2.TM_CCOEFF')
		plt.show()

	box_ret = [int(prev_box[0]-dely)+top_left[1], int(prev_box[1]-delx)+top_left[0], int(prev_box[0]-dely)+bottom_right[1], int(prev_box[1]-delx)+bottom_right[0]]
	#print(box_ret)
	return np.array(box_ret)


def tracker(prev_img, curr_img, track_interval = 5):
	#bounding boxes are currently in object_sv
	for key, value in object_sv.items():
		if value.f_count < track_interval:#normal update only using prev bounding box
			if value.occ_flag == False:
				kernel = kernel_cutter(prev_img, value.bounding_box)
				insane_box = insane_conv(curr_img, value.bounding_box, kernel, margin = 0.15)
				value.update(insane_box, kernel, track_interval)
			else:
				#do insane conv with bigger region !
				kernel = kernel_cutter(prev_img, value.last_seen_bb)
				insane_box = insane_conv(curr_img, value.last_seen_bb, kernel, margin = 0.3)
				valuese.update(insane_box, kernel, track_interval) #only upadtes bounding box 

		else:
			if value.occ_flag == False:
				insane_box1 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[0]], margin = 0.15)
				insane_box2 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[1]], margin = 0.15)
				insane_box3 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[2]], margin = 0.15)
				insane_box = (insane_box1 + insane_box2 + insane_box3) / 3
				kernel = kernel_cutter(curr_img, insane_box)
				value.update(insane_box, kernel, track_interval)
			else:
				#do insane conv with bigger region !
				insane_box1 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[0]], margin = 0.6)
				insane_box2 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[1]], margin = 0.6)
				insane_box3 = insane_conv(curr_img, value.bounding_box, value.kernels[value.sort_order[2]], margin = 0.6)
				insane_box = (insane_box1 + insane_box2 + insane_box3) / 3
				kernel = kernel_cutter(curr_img, insane_box)
				value.update(insane_box, kernel, track_interval) #only updates bounding box 

	#now that i have the new bouding boxes, let us check for occulsion!
	f, occ_keys, non_occ_keys = occulsion_detector(margin = 0.02, loc = True)
	for key in occ_keys:
		object_sv[key].flag_set(True)
	for key in non_occ_keys:
		object_sv[key].flag_set(False)


def print_state(image, input_image_name, plot = False):
	'''
	saves the image with tracking values
	'''
	out_boxes_temp = np.empty((0,4))
	out_ids_temp = np.array([])
	for key, value in object_sv.items():
		if value.occ_flag == False:
			out_boxes_temp = np.append(out_boxes_temp, [value.bounding_box], axis =0)
			out_ids_temp = np.append(out_ids_temp, [int(key[7:])], axis = 0)
		#out_ids_temp.append(int(key[7:]))
		#out_boxes_temp.append(value.bounding_box)
	#print(len(out_ids_temp))

	out_ids_temp = out_ids_temp.astype(int)
	draw_boxes_track(image, out_boxes_temp, out_ids_temp, track_colors)
	image.save(os.path.join("out_track_ic", input_image_name), quality=90)

	if(plot == True):
		#output_image = scipy.misc.imread(os.path.join("out_track_ic", input_image_name))
		#imshow(output_image)
		#plt.show()
		pass

#main

# Initiate a session
sess = K.get_session()

for frame_counter, frame in enumerate(video):
	#Get input image
	input_image_name = frame
	input_image = Image.open(VIDEO_DIR + input_image_name)
	width, height = input_image.size
	width = np.array(width, dtype=float)
	height = np.array(height, dtype=float)
	image_shape = (height, width)
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
	image_track, image_data_track = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
	if frame_counter > 0:
		prev_img, data = preprocess_image(VIDEO_DIR + video[frame_counter-1], model_image_size = (416, 416))

	if frame_counter == 0:
		#Define the graph 
		yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
		boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)

		#Run the session
		out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
		ppl_ind = np.argwhere(out_classes == 0) #for person class only!
		ppl_ind = np.squeeze(ppl_ind)
		#print(ppl_ind.shape)
		if ppl_ind.shape == ():
			ppl_ind = np.array([ppl_ind])
		out_scores = out_scores[ppl_ind]
		out_boxes = out_boxes[ppl_ind]
		out_classes = out_classes[ppl_ind]
		draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
		image.save(os.path.join("out", input_image_name), quality=90)

		#run yolo only once
		initialize_tracker(out_boxes)
		print_state(image_track, input_image_name)

	else:	
		tracker(prev_img, image_track, track_interval = 5)
		print_state(image_track, input_image_name)

	print(frame)

print("End of program")




