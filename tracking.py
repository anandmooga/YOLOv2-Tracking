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
threshlod = 0.975

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

def overlap_checker(bb_1, bb_2, margin = 1):
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
		print("Mapped suucesfully")

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


def occulsion_detector(out_boxes):
	occulsion_ids = []
	flag = False
	for i in range(len(out_boxes)):
		for j in range(i+1, len(out_boxes)):
			if overlap_checker(out_boxes[i], out_boxes[j]):
				flag = True
				occulsion_ids.append(i)
				occulsion_ids.append(j)
	return flag


def tracker(embeddings, out_boxes):
	'''
	Detects occulsion , new person entering and YOLO not able to recognise object
	Assumptions: 
		1. If the number of bouding boxes are same, then the same objects are detected as last time. We dont cosider the possibilty where one was lost and a new one entered.
		2. Assume that if object was at A location in prev frame, object will be at A + deltaA in next frame, which is true. 
		3. 

	Future work: make the entire trcker as an object 
	'''
	#checking for occulsion fault by using overlap. 

	if len(object_sv) == len(out_boxes):
		#check if occulsion occurs 
		if occulsion_detector(out_boxes):
			#deal with occulsion
			advanced_mapper(embeddings, out_boxes)
		else:
			#irrespectivive of weather faults state varible is empty or not, if the number is same I assume everything is normal
			simple_mapper(embeddings, out_boxes)
	else:
		#other 3 problems occulsion, lost track, new object
		advanced_mapper(embeddings, out_boxes)

	fault_flusher()

		
def advanced_mapper(embeddings, out_boxes):
	'''
	uses embeddings to idnetify objects, and tells us which are currently being tracked and which are problamatic 
	#let us assume that cosine distance works perfectly!
	'''
	global object_sv
	global object_count
	global pca

	len_objects_sv = len(object_sv)
	len_fault_sv = len(fault_sv)

	state_embedding_temp = np.empty((0,64))
	state_ids_temp = np.array([])
	for key, value in object_sv.items():
		state_embedding_temp = np.append(state_embedding_temp, [value.embedding], axis =0)
		state_ids_temp = np.append(state_ids_temp, [key], axis = 0)
	for key,value in fault_sv.items():
		state_embedding_temp = np.append(state_embedding_temp, [value.embedding], axis =0)
		state_ids_temp = np.append(state_ids_temp, [key], axis = 0)

	state_embedding_temp_tsne = pca.fit_transform(state_embedding_temp)
	#print(state_embedding_temp_tsne.shape, state_embedding_temp.shape)
	embeddings_tsne = pca.transform(embeddings)
	#print(embeddings_tsne.shape, embeddings.shape)
	#print(pca.explained_variance_ratio_)

	mat = cosine_similarity(state_embedding_temp_tsne, embeddings_tsne)
	#print(mat)
	mapping = np.argmax(mat, axis = 1)
	#print(mapping)
	conflict_mapping = np.argmax(mat, axis = 0)
	#print(conflict_mapping)

	full_object_set = set([i for i in range(len(embeddings))])
	new_objects_set = set(mapping)
	left_out = full_object_set - new_objects_set
	for i, ind in enumerate(mapping):
		
		if conflict_mapping[ind] != i :  #it means that this is either extra or new 
			mapping[i] = -1
			#now this could be in currently tracking or faults
			if i < len_objects_sv: #it means that this objects was being tracked and now is lost! so it has to be removed from object_sv and added to fault_sv
				#############try to map on basis of distance!
				'''
				#shoild use nearness mapper?
				npf , left_out = nearness_mapper(state_ids_temp[i], out_boxes, embeddings, left_out)
				if npf == False:
					fault_sv[state_ids_temp[i]] = object_sv[state_ids_temp[i]]
					del object_sv[state_ids_temp[i]]
				'''
				fault_sv[state_ids_temp[i]] = object_sv[state_ids_temp[i]]
				del object_sv[state_ids_temp[i]]

				#no need to update the object as it should save the last embeddings data of when it was being tracked
			else: #this means that the object was in fault and now is still in fault, counter flush to be implemeneted 
				pass

		else: #it has been mapped succesfully !
			#new_objects_set.add(ind) 
			#if mat[i][ind] < threshlod:
			#	mapping[i] = -1
			#see if threshlod required
			if i < len_objects_sv: #it means object was being tracked as is still being tracked, so do an update 
				object_sv[state_ids_temp[i]].update(embeddings[ind], out_boxes[ind], True)
			else: #it means that object was in faults and has now been reidentified, so move from fault_sv to object_sv
				object_sv[state_ids_temp[i]] = fault_sv[state_ids_temp[i]]
				del fault_sv[state_ids_temp[i]]
				object_sv[state_ids_temp[i]].update(embeddings[ind], out_boxes[ind], True)

	#now to handle the unmapped new objects

	for ind in left_out:
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
	image.save(os.path.join("out_track_eb", input_image_name), quality=90)

	if(plot == True):
		output_image = scipy.misc.imread(os.path.join("out_track", input_image_name))
		imshow(output_image)
		plt.show()




#main

# Initiate a session
sess = K.get_session()
pca = PCA(n_components=5)

frame_counter = 0
for frame in video:
	#Get input image
	input_image_name = frame
	input_image = Image.open(VIDEO_DIR + input_image_name)
	width, height = input_image.size
	width = np.array(width, dtype=float)
	height = np.array(height, dtype=float)
	image_shape = (height, width)
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))
	image_track, image_data_track = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))

	#Define the graph 
	yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
	boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold = 0.45)
	image, image_data = preprocess_image(VIDEO_DIR + input_image_name, model_image_size = (416, 416))

	#Run the session
	out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
	ppl_ind = np.argwhere(out_classes == 0) #for humans only!
	ppl_ind = np.squeeze(ppl_ind)
	print(ppl_ind.shape)
	if ppl_ind.shape == ():
		ppl_ind = np.array([ppl_ind])
	out_scores = out_scores[ppl_ind]
	out_boxes = out_boxes[ppl_ind]
	out_classes = out_classes[ppl_ind]
	draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	image.save(os.path.join("out", input_image_name), quality=90)

	if len(ppl_ind) == 0:
		continue

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
		tracker(emb_list, out_boxes)
		print_state(image_track, input_image_name)

	print(frame)

	frame_counter += 1

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
	

