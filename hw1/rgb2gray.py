import sys
import numpy as np
import cv2
import time
from filter import JointBilateralFilter
from locMin import findLocalMinimum
from locMin import plotErr


start = time.time()

img = cv2.imread(sys.argv[1])

p = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

sigmaS = [1,2,3]
sigmaR = [0.05,0.1,0.2]

vote = {(i,j,k):0 for i in p for j in p for k in p if np.around(i+j+k,decimals = 1) == 1}

img = img.astype('int')

for ss in sigmaS:
	for sr in sigmaR:
		print('sigma_S =',ss,' ','sigma_R =',sr)
		bf = JointBilateralFilter(ss,sr,img,img)
		errs = {}
		for i in p:
			for j in p:
				gray = np.ones(img.shape)
				gray[:,:,0] = i
				gray[:,:,1] = j
				gray[:,:,2] = np.around(1-(i+j),decimals = 1)
				if np.around(i+j,decimals = 1) > 1:
					continue
				print('	(B,G,R) = (',i,',',j,',',np.around(1-(i+j),decimals = 1),')')
				guid = (np.sum(gray*img,axis = 2))
				guid = guid.astype('int')
				new_pic = JointBilateralFilter(ss,sr,guid,img).astype('int')
				#ref_pic = cv2.ximgproc.jointBilateralFilter(src=img.astype(np.uint8), joint=guid.astype(np.uint8), d=2*3*ss+1, sigmaColor=sr*255, sigmaSpace=ss)
				err = np.sum(abs(new_pic - bf))/(img.shape[0]*img.shape[1]*img.shape[2])
				#pics[(i,j,np.around(1-(i+j),decimals = 1))] = guid
				errs[(i,j,np.around(1-(i+j),decimals = 1))] = err
				#print('average error = ',np.sum(abs(ref_pic.astype('int') - new_pic.astype('int')))/(img.shape[0]*img.shape[1]*img.shape[2]))
				#print('variance = ',np.var(ref_pic.astype('int') - new_pic.astype('int')))
				print('	err = ',err)
		minList = findLocalMinimum(errs)
		plotErr(np.array(list(errs.keys())),np.array(list(errs.values())),ss,sr)
		for i in minList:
			print('local minimum =',i)
			vote[i] = vote[i] + 1

voteList = [(i,vote[i]) for i in vote]
voteList.sort(key = lambda tup: tup[1],reverse = True)
voteList = np.array(voteList)

optList = []

for i in voteList:
	if i[1] != 0 and len(optList) < 3:
		optList.append(i[0])

print('optimal parameter:')
for i in optList:
	print('(B,G,R) =',i)

np.save(str(sys.argv[1]) + '_voteList.npy',voteList)

end = time.time()

print('elapsed time: ' + str(end-start))