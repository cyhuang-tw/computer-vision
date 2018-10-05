import numpy as np
import cv2
import time
from filter import JointBilateralFilter

#Debug Mode
Debug = True

start = time.time()

img = cv2.imread('0a.png')

pics = {}

sigmaS = [1,2,3]
sigmaR = [0.05,0.1,0.2]
#p = [0,0.1,...,0.9,1]
p = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

img = img.astype('int')

test = JointBilateralFilter(1,0.1,img,img)
cv2.imwrite('test.jpg',test)

ref = cv2.ximgproc.jointBilateralFilter(src=img.astype(np.uint8),joint=img.astype(np.uint8),d = 2*3*1 + 1,sigmaColor = 0.1*255, sigmaSpace = 1)
cv2.imwrite('ref.jpg',ref)

print('error = ',np.sum(abs(ref.astype('int') - test.astype('int')))/(img.shape[0]*img.shape[1]*img.shape[2]))

for ss in sigmaS:
	for sr in sigmaR:
		print('sigma_S = ',ss,' ','sigma_R = ',sr)
		for i in p:
			for j in p:
				gray = np.ones(img.shape)
				gray[:,:,0] = i
				gray[:,:,1] = j
				gray[:,:,2] = 1 - (i+j)
				if i+j > 1:
					continue
				print('(B,G,R) = (',i,',',j,',',np.around(1-(i+j),decimals = 1),')')
				guid = (np.sum(gray*img,axis = 2))
				guid = guid.astype('int')
				new_pic = JointBilateralFilter(ss,sr,guid,img).astype('uint8')
				ref_pic = cv2.ximgproc.jointBilateralFilter(src=img.astype(np.uint8), joint=guid.astype(np.uint8), d=2*3*ss+1, sigmaColor=sr*255, sigmaSpace=ss)
				pics[(i,j)] = new_pic

				#print('average error = ',np.sum(abs(ref_pic.astype('int') - new_pic.astype('int')))/(img.shape[0]*img.shape[1]*img.shape[2]))
				print('variance = ',np.var(abs(ref_pic.astype('int') - new_pic.astype('int'))))

end = time.time()

print('elapsed time: ' + str(end-start))