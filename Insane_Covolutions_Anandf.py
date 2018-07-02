import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Insane_conv/frame_123.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

template = cv2.imread('Insane_conv/frame_122.jpg')
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

method = eval('cv2.TM_CCOEFF')
# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
print(top_left, bottom_right)
cv2.rectangle(img,top_left, bottom_right, 255, 5)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.TM_CCOEFF')
plt.show()

