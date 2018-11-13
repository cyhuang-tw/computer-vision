import os
import sys
import numpy as np
from numpy import linalg as LA
import cv2
from PCAnLDA import PCA
from PCAnLDA import rebuildImg
from sklearn.manifold import TSNE
import matplotlib.pylab as plt 
import matplotlib as mpl
import matplotlib.cm as cm

##############################

pNum = 40
trainNum = 7
testNum = 3

height = 56
width = 46
nDim = height * width

eigVecNum = 279
projDim = 100

##############################
path = sys.argv[1]
fileList = os.listdir(path)
fileList.sort()

data = np.zeros((nDim,len(fileList)))
test = np.zeros((nDim,len(fileList)))

i = 0
j = 0

for file in fileList:
	fileName = os.path.splitext(file)[0]
	img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
	img = img.flatten()
	if int(fileName.split('_')[1]) <= 7:
		data[:,i] = img
		i = i + 1
	else:
		test[:,j] = img
		j = j + 1


data = data[:,:i]
test = test[:,:j]

eigenVals, eigenVecs, mean = PCA(data)

'''
meanFace = mean
meanFace = np.around(meanFace.reshape(height,width))
cv2.imwrite('meanFace.png',meanFace.astype(np.uint8))
'''
'''
idx = 1
for face in eigenVecs.T:
	curFace = face
	curFace = curFace - np.min(curFace)
	curFace = curFace / np.max(curFace)
	curFace = curFace * 255
	curFace = np.around(curFace.reshape(height,width))
	cv2.imwrite('eigenFace_' + str(idx) + '.png',curFace.astype(np.uint8))
	idx += 1
'''
'''
img = cv2.imread(os.path.join(path,'8_6.png'),cv2.IMREAD_GRAYSCALE)

newImg = rebuildImg(img,eigenVecs,mean,eigVecNum)
newImg = np.around(newImg.reshape(height,width))

MSE = np.sum(np.square(img.astype(float) - newImg.astype(float)))/(img.shape[0]*img.shape[1])

print('Mean Square Error:',MSE)

cv2.imwrite('8_6_' + str(eigVecNum) + '.png',newImg.astype(np.uint8))
'''

'''
pVals = (eigenVecs[:,:projDim]).T @ (test - np.repeat(mean,pNum*testNum,axis = 1))

X = TSNE(n_components=2,perplexity=50).fit_transform(pVals.T)

plt.title("PCA scattering (DIM = 100)")
for i in range(pNum):
	plt.scatter(X[i*testNum:(i+1)*testNum,0],X[i*testNum:(i+1)*testNum,1])
plt.show()
'''
testImg = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)

newImg = rebuildImg(testImg,eigenVecs,mean,eigVecNum)
newImg = np.around(newImg.reshape(height,width))

cv2.imwrite(sys.argv[3],newImg.astype(np.uint8))