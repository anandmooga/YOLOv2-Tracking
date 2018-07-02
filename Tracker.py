# import the needed modules
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
from scipy.spatial import distance
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import cv2

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.models import load_model

from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, draw_boxes_track
from yad2k.models.keras_yolo import yolo_head, yolo_eval
from sklearn.decomposition import PCA

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
threshold = 0.95

object_sv = {} #key is the identifier and value is list of some bb and embedding, has only non faulty objects !
object_count = 0


fault_sv = {} #key is identifier and value is embedding at time of fault identification, has only faulty objects
fault_count = 0

#keep track of 10 frames, if more than 10, delete

class object_v():
	'''
	Object that represents an object to be tracked
	'''
	def __init__(self, embedding, bounding_box, key):
		self.embedding = embedding
		self.bounding_box = bounding_box
		self.embedding_prev = embedding
		self.bounding_box_prev = bounding_box
		self.flag = False
		self.f_count = 0
		self.insnae_count = 0
		self.key = key
		self.bdist = -1 
		self.bdist_prev = -1

	def update(self, embedding, bounding_box, flag = True):
		self.embedding_prev = self.embedding
		self.embedding = embedding
		self.bounding_box_prev = self.bounding_box
		self.bounding_box = bounding_box
		self.flag = flag
		self.f_count = 0
		self.insnae_count = 0
		self.bdist_prev = self.bdist
		self.bdist = distance.cdist(self.bounding_box[np.newaxis,:], self.bounding_box_prev[np.newaxis, :])[0][0]
		'''
		if bdist > 2.5*bdist_prev:
			#wrong mapping dude
			return False
		else:
			return True
		'''

	def insane_update(self, bounding_box):
		self.bounding_box_prev = self.bounding_box
		self.bounding_box = bounding_box
		self.insnae_count += 1
		#print("Insane",self.bounding_box[np.newaxis,:])
		#print("Insane",self.bounding_box_prev[np.newaxis,:])
		#print(distance.cdist(self.bounding_box[np.newaxis,:], self.bounding_box_prev[np.newaxis, :]))
		self.bdist_prev = self.bdist
		self.bdist = distance.cdist(self.bounding_box[np.newaxis,:], self.bounding_box_prev[np.newaxis, :])[0][0]

	def update_fault(self, flag =False):
		self.flag = flag
		self.f_count += 1
		self.insnae_count = 0
		self.bdist_prev = 999999
		return self.f_count


def initialize_tracker(embeddings, out_boxes):
	'''
	It creates seprate objects for each object identified in the image. called only once in the begnning
	'''
	global object_sv
	global object_count
	for i in range(len(embeddings)):
		object_sv["object_" + str(object_count)] = object_v(embeddings[i], out_boxes[i], "object_" + str(object_count))
		object_count += 1


def overlap_checker(bb_1, bb_2, margin = 1.05):
	'''
	[0,1,2,3] are the values of bb, the the bounding box is (0,1)------------(2,1)
															  |                |
														  	  |                |
								  	  						(0,3)------------(2,3)
	'''
	if bb_1[0] > bb_2[2]*margin or bb_1[2] < bb_2[0]/margin:
		return False
	if bb_1[1] > bb_2[3]*margin or bb_1[3] < bb_2[1]/margin:
		return False
	return True

def occulsion_detector(out_boxes, loc = False):
	occulsion_ids = []
	flag = False
	for i in range(len(out_boxes)):
		for j in range(i+1, len(out_boxes)):
			if overlap_checker(out_boxes[i], out_boxes[j]):
				flag = True
				occulsion_ids.append(i)
				occulsion_ids.append(j)
	if loc == False:
		return flag
	else:
		return (flag, set(occulsion_ids))

#not used
def nearness_mapper(key, out_boxes, embeddings, left_out):
	'''
	Takes in one object and a list of bounding boxes and embedings, finds the corresponding left out set of new boxes
	'''
	flag = False
	
	embeddings_cp = np.empty((0,64))
	for ind in range(len(embeddings)):
		embeddings_cp = np.append(embeddings_cp, [embeddings[ind]], axis = 0)

	dist1 = cosine_similarity(object_sv[key].embedding_prev[np.newaxis, :], embeddings_cp)
	ind1 = np.argmax(dist1, axis = 0)[0]

	for ind in left_out:
		if ind1 == ind :
			object_sv[key].update(embeddings[ind], out_boxes[ind], True)
			flag = True
			left_out.remove(ind)
			return (flag, left_out)
	return (flag, left_out)

def overlap_mapper(key, out_boxes, embeddings, left_out):
	'''
	Takes in one object and a list of bounding boxes and embedings, finds the corresponding left out set of new boxes
	'''
	flag = False
	for ind in left_out:
		if overlap_checker(object_sv[key].bounding_box, out_boxes[ind]):
			#if overlap is there that means its the same
			object_sv[key].update(embeddings[ind], out_boxes[ind], True)
			flag = True
			left_out.remove(ind)
			return (flag, left_out)

	return (flag, left_out)

#not used
def overlap_fault_mapper(key, out_boxes, embeddings, left_out):
	'''
	Takes in one object and a list of bounding boxes and embedings, finds the corresponding left out set of new boxes
	'''
	flag = False
	for ind in left_out:
		if overlap_checker(fault_sv[key].bounding_box, out_boxes[ind]):
			#if overlap is there that means faulty object reidentified !
			object_sv[key] =  fault_sv[key]
			#del fault_sv[key]
			object_sv[key].update(embeddings[ind], out_boxes[ind], True)
			flag = True
			left_out.remove(ind)
			return (flag, left_out)
	return (flag, left_out)


def insane_conv(prev_img, curr_img, prev_box, key, scale = 1.1, insane_thresh = 3):
	#first we will take a crop out of the current frame on basis of previous bb
	#then we will use insane conv using the prev bounding box precise filter! and get new bb !
	#insance conv kicks in when yolo fails !

	flag = False
	if object_sv[key].insnae_count >= insane_thresh:
		return (flag, [-1,-1,-1,-1])
	else:
		flag = True
		#object is a viable candidate for insane conv!
		prev_img_cp = np.array(prev_img)
		prev_img_cp = cv2.cvtColor(prev_img_cp,cv2.COLOR_BGR2GRAY)
		curr_img_cp = np.array(curr_img)
		curr_img_cp = cv2.cvtColor(curr_img_cp,cv2.COLOR_BGR2GRAY)

		kernel = prev_img_cp[int(prev_box[0]):int(prev_box[2]), int(prev_box[1]):int(prev_box[3])]
		#plt.imshow(kernel)
		#plt.show()
		image_crop = curr_img_cp[int(prev_box[0]/scale):int(prev_box[2]*scale), int(prev_box[1]/scale):int(prev_box[3]*scale)]
		#plt.imshow(image_crop)
		#plt.show()

		w, h = kernel.shape[::-1]
		method = eval('cv2.TM_CCOEFF')
		res = cv2.matchTemplate(image_crop,kernel,method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)
		box = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
		
		'''
		cv2.rectangle(image_crop,top_left, bottom_right, 255, 5)
		plt.subplot(121),plt.imshow(res,cmap = 'gray')
		plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(image_crop,cmap = 'gray')
		plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
		plt.suptitle('cv2.TM_CCOEFF')
		plt.show()
		'''
		#scale back boxes to orginal image!
		box_ret = [int(prev_box[0]/scale)+top_left[1], int(prev_box[1]/scale)+top_left[0], int(prev_box[0]/scale)+bottom_right[1], int(prev_box[1]/scale)+bottom_right[0]]
		return (flag, np.array(box_ret))

def fault_flusher(flush_count = 15):
	'''
	Flushes objects in fault_sv if they have not been reidentifed for a long time
	'''
	to_del = []
	for key,value in fault_sv.items():
		count = fault_sv[key].update_fault(False)
		if count >= flush_count:
			to_del.append(key)
	for key in to_del:
		del fault_sv[key]

def print_state(image, input_image_name, plot = False):
	'''
	saves the image with tracking values
	'''
	out_boxes_temp = np.empty((0,4))
	out_ids_temp = np.array([])
	for key, value in object_sv.items():
		out_boxes_temp = np.append(out_boxes_temp, [value.bounding_box], axis =0)
		out_ids_temp = np.append(out_ids_temp, [int(key[7:])], axis = 0)
		#out_ids_temp.append(int(key[7:]))
		#out_boxes_temp.append(value.bounding_box)
		
	out_ids_temp = out_ids_temp.astype(int)
	draw_boxes_track(image, out_boxes_temp, out_ids_temp, track_colors)
	image.save(os.path.join("out_track_ic", input_image_name), quality=90)

	if(plot == True):
		output_image = scipy.misc.imread(os.path.join("out_track_ic", input_image_name))
		imshow(output_image)
		plt.show()

def advanced_mapper(embeddings, out_boxes, prev_img, curr_img):
	'''
	uses embeddings to idnetify objects, and tells us which are currently being tracked and which are problamatic 

	'''
	#initilaiztion stuff
	global object_sv
	global object_count

	out_boxes_cp = np.copy(out_boxes)
	embeddings_cp = np.copy(embeddings)
	len_objects_sv = len(object_sv)
	len_fault_sv = len(fault_sv)

	#first we need to make sure overlap is not there between given bounding boxes 
	
	occ_flag, occ_boxes = occulsion_detector(out_boxes, True)
	#print("Occulsion stuff: ", occ_flag, occ_boxes)
	if occ_flag == True:
		new_boxes = set([i for i in range(len(out_boxes))]) - occ_boxes
	else:
		new_boxes = set([i for i in range(len(out_boxes))])
	#occ_boxes contains the indices of bouding boxes that are occluded and new_boxes contains good ones 

	unmapped_embeddings = np.empty((0, len(embeddings[0])))
	unmapped_embeddings1 = np.empty((0, len(embeddings[0])))
	new_unmapped_embeddings = np.empty((0, len(embeddings[0])))
	
	#print("Boxes which are going to be compared via nearness mapper: ",new_boxes)
	#now we will iterate through currently tracked objects and then try to map them to non occlusidng boxes
	unmapped_objects1 = []
	for key, value in object_sv.items():
		flag, new_boxes = overlap_mapper(key, out_boxes_cp, embeddings_cp, new_boxes)
		if flag == False:
			#objects needed to be mapped with embeddings! so store them
			unmapped_objects1.append(key)
			unmapped_embeddings1 = np.append(unmapped_embeddings1, [object_sv[key].embedding], axis =0)
		else:
			#objct has been mapped succesfully and the corresponding box removed from new_boxes
			pass


	#the reminder will be mapped using distance+threshold on that distance!
	if len(new_boxes) == 0:
		#no need to do distance mapping 
		unmapped_objects = unmapped_objects1
		unmapped_embeddings = unmapped_embeddings1
	else:
		dist_thresh = 1.2
		new_boxes_list = np.array(list(new_boxes))
		unmapped_boxes = out_boxes[new_boxes_list]
		print("In distance mapping,", new_boxes_list, unmapped_boxes)
		unmapped_objects = []
		for i, key in enumerate(unmapped_objects1):
			distances = np.append(distance.cdist(object_sv[key].bounding_box[np.newaxis,:], unmapped_boxes), distance.cdist(object_sv[key].bounding_box_prev[np.newaxis,:], unmapped_boxes), axis = 0)
			min_id = np.argmin(distances, axis = 1)[0]
			ind = min_id % len(unmapped_boxes)
			corresponding_box_id = new_boxes_list[ind]
			if object_sv[key].bdist == -1:
				#object just created
				object_sv[key].update(embeddings_cp[corresponding_box_id], out_boxes_cp[corresponding_box_id], True)
				new_boxes.remove(corresponding_box_id)
				unmapped_boxes = np.delete(unmapped_boxes, ind, axis = 0)
				if len(unmapped_boxes) == 0:
					break
			else:
				#alrady tracking
				if distances[min_id] < object_sv[key].bdist*dist_thresh:
					#this means that it was mapped succesfully as distance is lesser than a ceratin threshold based on previous data!
					object_sv[key].update(embeddings_cp[corresponding_box_id], out_boxes_cp[corresponding_box_id], True)
					new_boxes.remove(corresponding_box_id)
					unmapped_boxes = np.delete(unmapped_boxes, ind, axis = 0)
					if len(unmapped_boxes) == 0:
						break
				else:
					#this means that it was not mapped
					unmapped_objects.append(key)
					unmapped_embeddings = np.append(unmapped_embeddings, [object_sv[key].embedding], axis =0)



	#now i will have objects left over from the above 2 filteration steps in unmapped_objects, and the objects in fault_sv
	#I will have occluded boxes in occ_boxes  and some left over boxes after filtration in new_boxes
	#now whatever was left in new_boxes wasnt mapped to any of the exisitng boxes so embeddings wont help, it has to be a new object
	
	####add remainder of new_boxes as new objcets
	'''
	for ind in new_boxes:
		object_sv["object_" + str(object_count)] = object_v(embeddings[ind], out_boxes[ind], "object_" + str(object_count))
		object_count += 1
	'''

	#add objects in fault sv to unmapped_embeddings
	unmapped_faults = []
	for key,value in fault_sv.items():
		unmapped_faults.append(key)
		unmapped_embeddings = np.append(unmapped_embeddings, [fault_sv[key].embedding], axis =0)
		
	#Embedding and Insane logic!
	new_boxes = new_boxes|occ_boxes #just to reuse old code 
	len_unmapped_object_embeddings = len(unmapped_objects)

	if len(new_boxes) == 0:
		#it means that whatever bounding boxes existed were mapped! now the extra ones have to be handled
		#the ones that were being tracked are now to be put in faults and the ones in faults remain in faults
		for key in unmapped_objects: #its fine because its an empty list
			insane_flag , insane_box = insane_conv(prev_img, curr_img, object_sv[key].bounding_box, key)
			if insane_flag:
				object_sv[key].insane_update(insane_box)
			else:
				#insane conv has run for too long! objects to be added to faults 
				fault_sv[key] = object_sv[key]
				del object_sv[key]
			

	else:
		if len(unmapped_embeddings) == 0:
			#this means that all the objects were mapped and no faults curenlty exist
			#so the left over boxes should be put as new objects !
			for ind in new_boxes:
				#add the new boxes which are unmapped as new objects to be tracked
				object_sv["object_" + str(object_count)] = object_v(embeddings[ind], out_boxes[ind], "object_" + str(object_count))
				object_count += 1

		else:
			#some of the new boxes are yet to be mapped !
			new_boxes_list = list(new_boxes)
			for ind in new_boxes_list:
				new_unmapped_embeddings = np.append(new_unmapped_embeddings, [embeddings[ind]], axis =0)

			#pca = PCA(n_components=5)
			#unmapped_embeddings_tsne = pca.fit_transform(unmapped_embeddings)
			#new_unmapped_embeddings_tsne = pca.transform(new_unmapped_embeddings)

			#print(unmapped_embeddings.shape, new_unmapped_embeddings.shape)
			#print(unmapped_embeddings_tsne.shape, new_unmapped_embeddings_tsne.shape)

			mat = cosine_similarity(unmapped_embeddings, new_unmapped_embeddings)
			mapping = np.argmax(mat, axis = 1)
			conflict_mapping = np.argmax(mat, axis = 0)\

			mapping_back = [new_boxes_list[i] for i in mapping]
			mapping_back = set(mapping_back)
			object_create = new_boxes - mapping_back

			#print(mapping)

			for i, ind in enumerate(mapping):
				if conflict_mapping[ind] != i: #extra one 
					mapping[i] = -1
					if i < len_unmapped_object_embeddings: # it means that object was being tracked but now has lost track!
						insane_flag , insane_box = insane_conv(prev_img, curr_img, object_sv[unmapped_objects[i]].bounding_box, unmapped_objects[i])
						if insane_flag:
							object_sv[unmapped_objects[i]].insane_update(insane_box)
						else:
							#insane conv has run for too long! objects to be added to faults 
							fault_sv[unmapped_objects[i]] = object_sv[unmapped_objects[i]]
							del object_sv[unmapped_objects[i]]
						
					else: # it means that a previous fault is still not identified !
						pass	
				else: #mapping has occured !
					if i < len_unmapped_object_embeddings: # it was being tracked and is still being tracked
						object_sv[unmapped_objects[i]].update(embeddings[new_boxes_list[ind]], out_boxes[new_boxes_list[ind]], True)
					else: #it was in faults and has been reidentified!
						i = i-len_unmapped_object_embeddings
						object_sv[unmapped_faults[i]] = fault_sv[unmapped_faults[i]]
						del fault_sv[unmapped_faults[i]]
						object_sv[unmapped_faults[i]].update(embeddings[new_boxes_list[ind]], out_boxes[new_boxes_list[ind]], True)

			#handle new left out boxes and add them to objects to be tracked!

			for ind in object_create:
				#add the new boxes which are unmapped as new objects to be tracked
				object_sv["object_" + str(object_count)] = object_v(embeddings[ind], out_boxes[ind], "object_" + str(object_count))
				object_count += 1



def tracker(embeddings, out_boxes, prev_img, curr_img):
	'''
	Detects occulsion , new person entering and YOLO not able to recognise object
	Assumptions: 
		1. If the number of bouding boxes are same, then the same objects are detected as last time. We dont cosider the possibilty where one was lost and a new one entered.
		2. Assume that if object was at A location in prev frame, object will be at A + deltaA in next frame, which is true. 
		3. 

	Future work: make the entire trcker as an object 
	'''
	#checking for occulsion fault by using overlap. 
	#print("Objects in track: ", object_sv.keys())
	#print("Objects lost out of track", fault_sv.keys())

	advanced_mapper(embeddings, out_boxes, prev_img, curr_img)
	fault_flusher()


#main

# Initiate a session
sess = K.get_session()
#pca = PCA(n_components=5)

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
	#Define the graph 
	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))

	#Run the session
	out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
	ppl_ind = np.argwhere(out_classes == 0) #for humans only!
	ppl_ind = np.squeeze(ppl_ind)
	#print(ppl_ind.shape)
	if ppl_ind.shape == ():
		ppl_ind = np.array([ppl_ind])
	out_scores = out_scores[ppl_ind]
	out_boxes = out_boxes[ppl_ind]
	out_classes = out_classes[ppl_ind]
	draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	image.save(os.path.join("out", input_image_name), quality=90)

	#Get embeddings 
	out_boxes_emb = np.copy(out_boxes)
	activation_map = yolo_model_cut.predict(image_data)
	activation_shape = activation_map.shape  #N,W,H,C
	out_boxes_emb[:, [0,2]] *= (activation_shape[1]/image_shape[0])
	out_boxes_emb[:, [1,3]] *= (activation_shape[2]/image_shape[1])
	out_boxes_emb = out_boxes_emb.astype(int)
	z = 1
	emb_list = []
	for roi in out_boxes_emb:
		ch_list = []
		for channel in range(activation_shape[3]):
			ch_list.append(np.average(activation_map[0, roi[0]:roi[2], roi[1]:roi[3], channel]))
		embeddings.append(ch_list)
		emb_list.append(np.array(ch_list))
		names.append("f" + frame[5:10]+ "_" + class_names[out_classes[z-1]] + "_" +  str(z))
		#emb_list contains embeddings
		z += 1
	emb_list = np.array(emb_list)

	if frame_counter == 0:
		initialize_tracker(emb_list, out_boxes)
		print_state(image_track, input_image_name)
	else:	
		tracker(emb_list, out_boxes, prev_img, image_track)
		print_state(image_track, input_image_name)

	print(frame)

	#if frame == "frame_0005.jpg":
	#	break



import pandas as pd

dfe = pd.DataFrame(embeddings)
#print(dfe.head())
#dfe.to_csv("embeddings.tsv",sep = "\t", header = False, index = False)

dfn = pd.DataFrame(names)
#print(dfn.head())
#dfn.to_csv("objects.tsv",sep = "\t", header = False, index = False)

print(len(embeddings))
print(len(names))
print("END OF PROGRAM")
	