import numpy as np
import cv2

def genImg(img,bgr,fileName):
	gray = np.zeros(img.shape)
	gray[:,:,0] = bgr[0]
	gray[:,:,1] = bgr[1]
	gray[:,:,2] = bgr[2]

	result = np.sum(gray*img,axis = 2).astype('uint8')
	cv2.imwrite(fileName,result)	