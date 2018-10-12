import numpy as np
import cv2

i = 0.114
j = 0.587
k = 0.299

img = cv2.imread('0c.png')

gray = np.ones(img.shape)

gray[:,:,0] = i
gray[:,:,1] = j
gray[:,:,2] = k

guid = (np.sum(gray*img,axis = 2))

cv2.imwrite('0c_y.png',guid)