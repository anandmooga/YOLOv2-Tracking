# import the needed modules
'''
Improvements to be made:
	1. Gsoc box filtering should be overlap based!
	2. Square packing insted of rectange packing, i.e. better packing
	3. Permanence
	4. Better Gsoc boxes??
	5. What about completly stationary objects?
	6. Object occulsion is not a problem because if he's hidden, yolo cannot detect anyway

Add:
	1. Gaussian BLur
	2. accumulateWeighted

'''




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

def get_po(bb1, bb2):
	'''
	bb1 is the cut-out bounding box, 
	bb2 is the prediction of object detector

	returns percent overlap 
	'''
	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[1], bb2[1])
	y_top = max(bb1[0], bb2[0])
	x_right = min(bb1[3], bb2[3])
	y_bottom = min(bb1[2], bb2[2])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	#iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	iou = intersection_area / float(bb2_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou

def box_per(out_box, hpercent = 0.2, wpercent = 2*0.15):
	'''
	returns (y,x) to be added or subtracted !
	'''
	h = out_box[2] - out_box[0]
	w = out_box[3] - out_box[1]
	
	return(int(h*hpercent), int(w*wpercent))


def occulsion_detector(boxes, margin = 0.05, threshold = 0.4, flat = False):
	if margin != 0.0:
		#add the margin to the boxes
		for i in range(len(boxes)):
			box = boxes[i]
			dely, delx = box_per(box, margin, margin)
			boxes[i] = [box[0]-dely, box[1]-delx, box[2]+dely, box[3]+delx]

	occ_ids = []
	for i in range(len(boxes)):
		for j in range(i+1, len(boxes)):
			po = max(get_po(boxes[i], boxes[j]), get_po(boxes[j], boxes[i]))
			if po > threshold:
				occ_ids.append([i,j])
	if flat == False:
		return occ_ids
	else:
		return list(set(np.reshape(np.array(occ_ids), (-1))))

def area(box):
	return (box[2]-box[0])*float(box[3]-box[1])


class boundingBoxes():
	def __init__(self, boxes, pad = 4, margin = 0.0, threshold = 0.4):
		'''
		Used to store Bounding Boxes of previous 2 states
		boxes should be numpy array of shape [:, 4]
		'''
		self.boxes_prev = boxes
		self.boxes  = boxes
		self.pad = pad
		self.margin = margin
		self.threshold = threshold

	def update(self, boxes):
		self.boxes_prev = self.boxes
		self.boxes = boxes

		

	def reshaper(self, image_full, image_shape, to_plot = False):
		'''
		Returns cut and reshaped image in basis of prev bounding boxes
		Arguments that are returned :
			1. numpy image 
			2. pil image
			3. image shape
		Arguments that are stored for remapping:
			1. cut_edges //coordinates from where we cut in orig image 
			2. positions
			3. position_boxes
		'''
		cut_images = []
		cut_images_sizes = []
		cut_edges = []

		print("BOXES WE ARE GOING TO CUT WITH ! :")
		for roi in self.boxes:
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
			cut_out = np.pad(cut_out, [(self.pad,self.pad), (self.pad,self.pad), (0,0)], "constant")
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
		self.boxes = self.boxes[ind]
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

		self.cut_edges = cut_edges
		self.positions = positions
		self.position_boxes = position_boxes

		image_pil_reshaped = Image.fromarray(np.uint8(reshaped_image*255))
		return (reshaped_image[np.newaxis, :, :, :], image_pil_reshaped, image_shape_reshaped)

	def remapper(self, re_boxes):
		
		#finding the corresponding prev bounding box !
		## To do this we will map distance of current bounding boxes with the cut out box co-ordinates !
		re_boxes = np.array(re_boxes)
		good_ind = [i for i in range(len(re_boxes))]
		mapind = []
		for box in re_boxes:
			max_po = 0.0
			ind = -1
			for i , cut_outbox in enumerate(self.position_boxes):  
				po = get_po(cut_outbox, box)
				if po != 0.0 and po > max_po:
					#there is overlap
					max_po = po
					ind = i
			mapind.append(ind)

		# mapping back the bounding boxes
		to_del = []
		new_boxes = []
		for ind_curr, ind_prev in enumerate(mapind):
			if ind_prev != -1:
				#now to map back
				roi = self.cut_edges[ind_prev]
				h = re_boxes[ind_curr][2] - re_boxes[ind_curr][0]
				w = re_boxes[ind_curr][3] - re_boxes[ind_curr][1]
				offsets = re_boxes[ind_curr][0:2] - self.positions[ind_prev] - np.array([self.pad, self.pad])
				print(roi[0], roi[1])
				print(offsets, h, w)
				mapped_back_box = np.array([roi[0]+int(offsets[0]), roi[1]+int(offsets[1]), roi[0]+int(offsets[0]+h), roi[1]+int(offsets[1]+w)])
				new_boxes.append(mapped_back_box)
			else:
				#box is nonsense
				to_del.append(ind_curr)
				pass

		new_boxes = np.array(new_boxes)
		
		mapind = np.delete(mapind, to_del, axis = 0).tolist()
		re_boxes = np.delete(re_boxes, to_del, axis = 0)
		good_ind = np.delete(good_ind, to_del, axis = 0).tolist()

		occ_ids = occulsion_detector(new_boxes, self.margin, self.threshold)
		
		to_del = []
		for ids in occ_ids:
			if mapind[ids[0]] != mapind[ids[1]]:
				#same object detected again!
				if area(new_boxes[ids[0]]) >= area(new_boxes[ids[1]]):
					to_del.append(ids[1])
		to_del = list(set(to_del))

		#delete reppeting boxes
		new_boxes = np.delete(new_boxes, to_del, axis = 0)
		good_ind = np.delete(good_ind, to_del, axis = 0)

		return (new_boxes, good_ind)


parser = argparse.ArgumentParser()
parser.add_argument('-vpath', action = 'store', dest = 'video_path', help = 'Read frame path', default = 'Video_tsne/')

results = parser.parse_args()
VIDEO_DIR = results.video_path 

frames = os.listdir(os.path.join(os.getcwd(), VIDEO_DIR))
frames.sort()

fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC(mc = 1, nSamples = 20, replaceRate = 0.2, propagationRate = 0.2, hitsThreshold = 16, noiseRemovalThresholdFacBG = 0.002, noiseRemovalThresholdFacFG = 0.004)
	

#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolov2_fc_anchors.txt")
colors = generate_colors(class_names)

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolov2_fc.h5")


sess = K.get_session()

'''
1st iteration init the boundingBoxes
2nd interation :
	generate image data
	call reshaper
	use yolo
	call remapper 
	update boxes
'''

z=0
init_flag = False
while(1):
	#reading the image
	frame = cv2.imread(os.path.join(VIDEO_DIR, frames[z]))
	z +=1

	#foreground mask
	img = fgbg.apply(frame)	
	kernel = np.ones((5,5),np.uint8)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	if z > 5:
		if init_flag == False:
			contours = cv2.findContours(img, cv2.RETR_LIST, 2)
			cnt = contours[1]
			count = 0
			boxes_bg = []
			for i in cnt:
				x,y,w,h = cv2.boundingRect(i)
				if w > 15 and h > 30:
					count += 1
					boxes_bg.append([y,x,y+h,x+w])
					#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			boxes_bg = np.array(boxes_bg)
			init_flag = True
			small_yolo = boundingBoxes(boxes_bg)

		else:
			#read the image
			input_image_name = frames[z]
			#Obtaining the dimensions of the input image
			input_image = Image.open(VIDEO_DIR + input_image_name)
			width, height = input_image.size
			width = np.array(width, dtype=float)
			height = np.array(height, dtype=float)
			image_shape = (height, width)
			image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))

			#calling reshaper
			rimage_data, rimage, rimage_shape = small_yolo.reshaper(np.array(image), image_shape)

			#use yolo
			yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
			boxes, scores, classes = yolo_eval(yolo_outputs, rimage_shape, score_threshold = 0.45)

			out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:rimage_data,K.learning_phase(): 0})
			print("Initial bounding box: \n", out_boxes)
			print("Initial bounding box shapes: \n", out_boxes[:, 2:4]-out_boxes[:, 0:2])
			draw_boxes(rimage, out_scores, out_boxes, out_classes, class_names, colors)
			rimage.save(os.path.join("out", input_image_name), quality=90)

			#calling remapper
			out_boxes, good_ind = small_yolo.remapper(out_boxes)
			draw_boxes(image, out_scores[good_ind], out_boxes, out_classes[good_ind], class_names, colors)
			image.save(os.path.join("out_track", input_image_name), quality=90)

			#getting new bb's from gsoc
			contours = cv2.findContours(img, cv2.RETR_LIST, 2)
			cnt = contours[1]
			count = 0
			boxes_bg = []
			for i in cnt:
				x,y,w,h = cv2.boundingRect(i)
				if w > 15 and h > 30:
					count += 1
					boxes_bg.append([y,x,y+h,x+w])
					#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			boxes_bg = np.array(boxes_bg)
			small_yolo.update(boxes_bg)

			
print("End of program")







