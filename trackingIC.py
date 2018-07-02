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
print(class_names)
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
		self.key = key

	def update(self, embedding, bounding_box, flag = True):
		self.embedding_prev = self.embedding
		self.embedding = embedding
		self.bounding_box_prev = self.bounding_box
		self.bounding_box = bounding_box
		self.flag = flag
		self.f_count = 0

	def insane_update(self, bounding_box):
		self.bounding_box_prev = self.bounding_box
		self.bounding_box = bounding_box

	def update_fault(self, flag =False):
		self.flag = flag
		self.f_count += 1
		return self.f_count

	def reidentification():
		pass


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


def simple_mapper(embeddings, out_boxes):
	'''
	Maps the relation objects in 2 difrrent frames, this is operating under the assuption 1 and 2.
	So, i will have same number of objects which are not flagged in state variable as the number of bounding boxes
	When simple mapper is working there is no occulsion, so when i compare prev boxes with new boxes , the ones that overlap are the same objects 
	'''
	assert len(object_sv) == len(out_boxes) , "number of bounding boxes should be same as number of objects currently tracking"
	out_boxes_cp = np.copy(out_boxes)
	embeddings_cp = np.copy(embeddings)
	problems = np.array([])
	
	for key,value in object_sv.items():
		to_del = False
		for i, box in enumerate(out_boxes_cp):
			#here is where the logic to compare is used
			if overlap_checker(value.bounding_box, box) or overlap_checker(value.bounding_box_prev, box):   #an or can be added to check one bb prev as well 
				value.update(embeddings_cp[i], box, False)
				to_del = i
				break
		if to_del == False:
			#no match found, we will compare in the end
			problems = np.append(problems, [key], axis =0)
		else:
			out_boxes_cp = np.delete(out_boxes_cp, to_del, axis = 0)
			embeddings_cp = np.delete(embeddings_cp, to_del, axis = 0)

	#if there were problems !
	#problem would be that more than one boxes did not have overlap
	#check if problems is not empty and then associate on basis of distance 
	if problems.size != 0:
		#print(problems.size)
		for key in problems:
			#print(object_sv[key].bounding_box)
			#print("CHECK ", distance.cdist(object_sv[key].bounding_box[np.newaxis,:], out_boxes_cp) )
			distances = np.append(distance.cdist(object_sv[key].bounding_box[np.newaxis,:], out_boxes_cp), distance.cdist(object_sv[key].bounding_box_prev[np.newaxis,:], out_boxes_cp), axis = 0)
			min_id = np.argmin(distances, axis = 1)[0]
			corresponding_box_id = min_id % len(out_boxes_cp)
			#corresponding_box_id = np.argmin(distance.cdist(object_sv[key].bounding_box[np.newaxis,:], out_boxes_cp), axis = 1)[0]
			object_sv[key].update(embeddings_cp[corresponding_box_id], out_boxes_cp[corresponding_box_id], False)
			out_boxes_cp = np.delete(out_boxes_cp, corresponding_box_id, axis = 0)
			embeddings_cp = np.delete(embeddings_cp, corresponding_box_id, axis = 0)
	if(out_boxes_cp.size == 0):
		#print("Mapped suucesfully")
		pass


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

	if len(object_sv) == len(out_boxes):
		#check if occulsion occurs 
		if occulsion_detector(out_boxes):
			#deal with occulsion
			advanced_mapper(embeddings, out_boxes, prev_img, curr_img)
		else:
			#irrespectivive of weather faults state varible is empty or not, if the number is same I assume everything is normal
			simple_mapper(embeddings, out_boxes)
	else:
		#other 3 problems occulsion, lost track, new object
		advanced_mapper(embeddings, out_boxes, prev_img, curr_img)

	fault_flusher()

def nearness_mapper_old(key, out_boxes, embeddings, left_out):
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

def nearness_mapper(key, out_boxes, embeddings, left_out):
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

def nearness_fmapper(key, out_boxes, embeddings, left_out):
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

def insane_conv(prev_img, curr_img, prev_box, scale = 1.2):
	#first we will take a crop out of the current frame on basis of previous bb
	#then we will use insane conv using the prev bounding box precise filter! and get new bb !
	#insance conv kicks in when yolo fails !

	prev_img_cp = np.array(prev_img)
	prev_img_cp = cv2.cvtColor(prev_img_cp,cv2.COLOR_BGR2GRAY)
	#plt.imshow(prev_img_cp)
	#print(prev_img_cp.shape)
	#plt.show()
	curr_img_cp = np.array(curr_img)
	curr_img_cp = cv2.cvtColor(curr_img_cp,cv2.COLOR_BGR2GRAY)
	#plt.imshow(curr_img_cp)
	plt.show()

	kernel = prev_img_cp[int(prev_box[0]):int(prev_box[2]), int(prev_box[1]):int(prev_box[3])]
	#print(int(prev_box[0]), int(prev_box[2]), int(prev_box[1]), int(prev_box[3]))
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
	box_ret = [int(prev_box[0]/scale)+top_left[1], int(prev_box[1]/scale)+top_left[0], int(prev_box[0]/scale)+bottom_right[1], int(prev_box[1]/scale)+bottom_right[0]]
	return box_ret

	
def advanced_mapper(embeddings, out_boxes, prev_img, curr_img):
	'''
	uses embeddings to idnetify objects, and tells us which are currently being tracked and which are problamatic 
	#let us assume that cosine distance works perfectly!
	'''
	global object_sv
	global object_count

	out_boxes_cp = np.copy(out_boxes)
	embeddings_cp = np.copy(embeddings)
	len_objects_sv = len(object_sv)
	len_fault_sv = len(fault_sv)

	unmapped_embeddings = np.empty((0, len(embeddings[0])))
	new_unmapped_embeddings = np.empty((0, len(embeddings[0])))

	#first we need to make sure overlap is not there between given bounding boxes 
	
	occ_flag, occ_boxes = occulsion_detector(out_boxes, True)
	#print("Occulsion stuff: ", occ_flag, occ_boxes)
	if occ_flag == True:
		new_boxes = set([i for i in range(len(out_boxes))]) - occ_boxes
	else:
		new_boxes = set([i for i in range(len(out_boxes))])

	#print("Boxes which are going to be compared via nearness mapper: ",new_boxes)

	#now we will iterate through currently tracked objects and then try to map them to non occlusidng boxes
	unmapped_objects = []
	for key, value in object_sv.items():
		flag, new_boxes = nearness_mapper(key, out_boxes_cp, embeddings_cp, new_boxes)
		if flag == False:
			#objects needed to be mapped with embeddings! so store them
			unmapped_objects.append(key)
			unmapped_embeddings = np.append(unmapped_embeddings, [object_sv[key].embedding], axis =0)
		else:
			#objct has been mapped succesfully and the corresponding box removed from new_boxes
			pass

	#same thing for faults
	unmapped_faults = []
	to_del = []
	for key,value in fault_sv.items():
		flag, new_boxes = nearness_fmapper(key, out_boxes_cp, embeddings_cp, new_boxes)
		if flag == False:
			#fault was not mapped back ! so store it 
			unmapped_faults.append(key)
			unmapped_embeddings = np.append(unmapped_embeddings, [fault_sv[key].embedding], axis =0)
		else:
			to_del.append(key)
			#fault was reidentifeid, and removed from fault_sv and added and updated in object_sv, 
			pass
	for key in to_del:
		del fault_sv[key]

	#now we have some objects left in unmapped_object, unmapped_faults and also some new boxes that havent been mapped

	#now in-order to help reidentify objects lost we will use Insane Convolutions!
	'''
	for key in unmapped_objects:
		print(key)
		insane_box = insane_conv(prev_img, curr_img, object_sv[key].bounding_box)
		object_sv[key].insane_update(insane_box)
	unmapped_objects = []
	'''

	len_unmapped_object_embeddings = len(unmapped_objects)
	new_boxes = new_boxes|occ_boxes #unioin of left out and occluded boxes!
	#print("Mapping via embeddings : ", new_boxes)

	#print("unmapped objects = ",unmapped_objects)
	#print("unmapped faults = ", unmapped_faults)

	#print("Objects in track at time of embedding: ", object_sv.keys())
	#print("Objects lost out of track at time of embedding", fault_sv.keys())


	if len(new_boxes) == 0:
		#it means that whatever bounding boxes existed were mapped! now the extra ones have to be handled
		#the ones that were being tracked are now to be put in faults and the ones in faults remain in faults
		for key in unmapped_objects: #its fine because its an empty list
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
						#use neasrness mapper old one !
						npf , object_create = nearness_mapper_old(unmapped_objects[i], out_boxes_cp, embeddings_cp, object_create)
						insane_conv(prev_img, curr_img, object_sv[unmapped_objects[i]].bounding_box)
						if npf == False:
							insane_box = insane_conv(prev_img, curr_img, object_sv[unmapped_objects[i]].bounding_box)
							object_sv[unmapped_objects[i]].insane_update(insane_box)
							#fault_sv[unmapped_objects[i]] = object_sv[unmapped_objects[i]]
							#del object_sv[unmapped_objects[i]]
						
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


		
def fault_flusher():
	to_del = []
	for key,value in fault_sv.items():
		count = fault_sv[key].update_fault(False)
		if count >= 100:
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
		output_image = scipy.misc.imread(os.path.join("out_track", input_image_name))
		imshow(output_image)
		plt.show()




#main

# Initiate a session
sess = K.get_session()
pca = PCA(n_components=5)


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


'''
	if frame_counter == 0:
		initialize_tracker(emb_list, out_boxes)
		print_state(image_track, input_image_name)
	else:
		if fault_detector:
			if flag_comparartor:
				simple_mapper
			else:


		else:	
			simple_mapper(emb_list, out_boxes)
			print_state(image_track, input_image_name)


def tracker(embeddings, out_boxes):
	
	Detects occulsion , new person entering and YOLO not able to recognise object
	Assumptions: 
		1. If the number of bouding boxes are same, then the same objects are detected as last time. We dont cosider the possibilty where one was lost and a new one entered.
		2. Assume that if object was at A location in prev frame, object will be at A + deltaA in next frame, which is true. 
		3. 

	Future work: make the entire trcker as an object 
	
	#checking for occulsion fault by using overlap. 

	if len(object_sv) == len(out_boxes):

		#check if occulsion occurs 
		if occulsion_detector(out_boxes):
			#deal with occulsion
			pass
		else:
			#irrespectivive of weather faults state varible is empty or not, if the number is same I assume everything is normal
			simple_mapper(embeddings, out_boxes)

	else:
		if occulsion_detector(out_boxes):
			#deal with occulsion
			pass

		#checking for loose of track 
		if len(object_sv) > len(out_boxes):
			pass
			#lost track

		#checking for new person
		if len(out_boxes) > len(object_sv):
			#new person
			pass 

'''
print("END OF PROGRAM")
	

