import numpy as np
import cv2


def JointBilateralFilter(ss,sr,guid,target):

#sr is normalized (0~255 -> 0~1)

	r = 3*ss
	w = 2*r + 1 #window size
	sr = sr*255

	result = np.zeros(target.shape)

	#create the window for spatial part
	s_tmpX = np.arange(-r,r+1)
	s_tmpX = s_tmpX.reshape(1,2*r+1)
	s_tmpX = s_tmpX.repeat(2*r+1,axis=0)
	s_tmpY = s_tmpX.transpose()
	s = (-1)*(s_tmpX*s_tmpX + s_tmpY*s_tmpY)/(2*(ss**2))
	window_s = np.exp(s)
	window_s = window_s.reshape((w,w,1))

	#padding for target and guidance pic
	img = cv2.copyMakeBorder(target,r,r,r,r,cv2.BORDER_REFLECT)
	guid_p = cv2.copyMakeBorder(guid,r,r,r,r,cv2.BORDER_REFLECT)

	#filtering
	#for BW pics
	if len(guid_p.shape) == 2:
		for i in range(guid.shape[0]):
			for j in range(guid.shape[1]):
				#create the window for color part
				cur_window = img[i:(i+w),j:(j+w)]
				window_tmp = guid_p[i:(i+w),j:(j+w)]
				window_r = window_tmp - window_tmp[r][r]
				window_r = np.exp((-1)*(window_r*window_r)/(2*(sr**2)))
				window_r = window_r.reshape((w,w,1))

				result[i][j] = (np.sum(window_r*window_s*cur_window,axis=(0,1)))/(np.sum(window_s*window_r))
	else:
		#for color pics
		for i in range(guid.shape[0]):
			for j in range(guid.shape[1]):
				#create the window for color part
				cur_window = img[i:(i+w),j:(j+w)]
				window_tmp = guid_p[i:(i+w),j:(j+w)]
				window_r = window_tmp - window_tmp[r][r]
				window_r = np.exp((-1)*(window_r*window_r)/(2*(sr**2)))
				window_r = (window_r[:,:,0]*window_r[:,:,1]*window_r[:,:,2]).reshape((w,w,1))

				result[i][j] = (np.sum(window_r*window_s*cur_window,axis=(0,1)))/(np.sum(window_s*window_r))
				
	return result