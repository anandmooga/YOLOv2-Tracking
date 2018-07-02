import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Video_tsne/frame_0017.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = img.copy()
template = cv2.imread('Video_tsne/frame_0001_kernel.jpg')
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
template2 = cv2.imread('human_body_kernel1.jpg')
template2 = cv2.cvtColor(template2,cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF'] #best method 
for meth in methods:
	method = eval(meth)
	for i in range(1):
		img = img2.copy()
		i = i + 1
		# Apply template Matching
		res = cv2.matchTemplate(img,template,method)

		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		print("i: ", i)
		print(min_val, max_val)
		# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
			top_left = min_loc
		else:
			top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)
		cv2.rectangle(img,top_left, bottom_right, 255, 5)
		plt.subplot(121),plt.imshow(res,cmap = 'gray')
		plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(img,cmap = 'gray')
		plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
		plt.suptitle(meth)
		plt.show()
