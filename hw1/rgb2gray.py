import sys
import os
import numpy as np
import cv2
import time
from filter import JointBilateralFilter
from locMin import findLocalMin
from genImg import genImg


start = time.time()

img = cv2.imread(sys.argv[1])

name = os.path.splitext(sys.argv[1])

p = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

sigmaS = [1,2,3]
sigmaR = [0.05,0.1,0.2]

vote = {(i,j,k):0 for i in p for j in p for k in p if np.around(i+j+k,decimals = 1) == 1}

img = img.astype('int')

genImg(img,(0.114,0.587,0.299),name[0] + '_y' + name[1])

for ss in sigmaS:
	for sr in sigmaR:
		print('sigma_S =',ss,' ','sigma_R =',sr)
		bf = JointBilateralFilter(ss,sr,img,img).astype('int')
		cost = {}
		for i in p:
			for j in p:
				k = np.around(1-(i+j),decimals = 1)
				if k < 0:
					continue
				gray = np.zeros(img.shape)
				gray[:,:,0] = i
				gray[:,:,1] = j
				gray[:,:,2] = k
				print('  (B,G,R) = (',i,',',j,',',k,')')
				guid = (np.sum(gray*img,axis = 2)).astype('int')
				new_pic = JointBilateralFilter(ss,sr,guid,img).astype('int')
				cur_cost = np.sum(abs(new_pic - bf))/(img.shape[0]*img.shape[1]*img.shape[2])
				cost[(i,j,k)] = cur_cost
				print('    cost =',cur_cost)
		minList = findLocalMin(cost)
		print('local minima:')
		for i in minList:
			print(i)
			vote[i] = vote[i] + 1

voteList = [(i,vote[i]) for i in vote]
voteList.sort(key = lambda tup: tup[1],reverse = True)
voteList = np.array(voteList)

optList = []

for p in voteList:
	if p[1] != 0 and (len(optList) < 3 or p[1] == optList[-1][1]):
		optList.append(p)

print('optimal parameters:')
for i in range(len(optList)):
	print('(B,G,R) =',optList[i][0])
	genImg(img,optList[i][0],name[0] + '_y' + str(i+1) + name[1])

end = time.time()

print('elapsed time: ' + str(end-start))