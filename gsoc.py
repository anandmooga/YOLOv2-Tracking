import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vpath', action = 'store', dest = 'video_path', help = 'Read frame path', default = 'Video_tsne/')

results = parser.parse_args()
VIDEO_DIR = results.video_path 

frames = os.listdir(os.path.join(os.getcwd(), VIDEO_DIR))
frames.sort()

fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC(mc = 1, nSamples = 20, replaceRate = 0.1, propagationRate = 0.2, hitsThreshold = 16, noiseRemovalThresholdFacBG = 0.002, noiseRemovalThresholdFacFG = 0.004)
avg = cv2.imread(os.path.join(VIDEO_DIR, frames[0]))
avg = np.float32(avg)


z =0
while(1):
	#reading the image
	frame = cv2.imread(os.path.join(VIDEO_DIR, frames[z]))
	frame_all = cv2.imread(os.path.join(VIDEO_DIR, frames[z]))
	z +=1

	# background generation
	cv2.accumulateWeighted(frame,avg,0.1)
	res = cv2.convertScaleAbs(avg)

	#foreground mask
	img = fgbg.apply(frame)	

	kernel = np.ones((5,5),np.uint8)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	#close is just dilation followed by erosion

	img = cv2.GaussianBlur(img, (5,5), 0)



	if z > 5:
		contours = cv2.findContours(img, cv2.RETR_LIST, 2)
		cnt = contours[1]
		count = 0
		boxes_bg = []
		for i in cnt:
			x,y,w,h = cv2.boundingRect(i)
			cv2.rectangle(frame_all,(x,y),(x+w,y+h),(0,255,0),2)
			if w > 15 and h > 30:
				count += 1
				boxes_bg.append([y,x,y+h,x+w])
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		boxes_bg = np.array(boxes_bg)


	disp = np.hstack((frame_all, frame))
	cv2.imshow('GSOC', res)
	cv2.imshow("BB's", disp)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	if z == len(frames)-1:
		z =0

cv2.destroyAllWindows()
