import sys
import os
import numpy as np
import cv2
import numpy.linalg as LA
from sklearn.manifold import TSNE
import matplotlib.pylab as plt 
import matplotlib as mpl
import matplotlib.cm as cm
from PCAnLDA import PCA
from PCAnLDA import LDA

path = sys.argv[1]

fileList = os.listdir(path)

pNum = 40
trainNum = 7
testNum = 3

height = 56
width = 46
dimN = height * width
projDim = 30

tmpData = np.zeros((dimN,pNum,trainNum))
tmpTest = np.zeros((dimN,pNum,testNum))
data = np.zeros((dimN,pNum*trainNum))
test = np.zeros((dimN,pNum*testNum))


for file in fileList:
  fileName = (os.path.splitext(file))[0]
  label = list(map(int,fileName.split('_')))
  img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE).flatten()
  if label[1] <= 7:
    tmpData[:,label[0]-1,label[1]-1] = img
  else:
    tmpTest[:,label[0]-1,label[1]-trainNum-1] = img

data = tmpData.reshape(dimN,pNum*trainNum)
test = tmpTest.reshape(dimN,pNum*testNum)

eigenVals, eigenVecs, mean = PCA(data)

pVals = (eigenVecs).T @ (data - np.repeat(mean,pNum*trainNum,axis = 1))

wVals, wVecs = LDA(pVals,pNum,trainNum)

eigMat = eigenVecs[:,:(pNum*trainNum - pNum)]

fisherFaces = eigMat @ wVecs

face = fisherFaces[:,0]
face = face - np.min(face)
face = face / np.max(face)
face = face * 255
face = face.reshape(height,width)
cv2.imwrite(sys.argv[2],face.astype(np.uint8))

'''
count = 0
for face in fisherFaces.T:
  curFace = face
  curFace = curFace - np.min(curFace)
  curFace = curFace / np.max(curFace)
  curFace = curFace*255
  curFace = np.around(curFace)
  curFace = curFace.reshape(height,width)
  curFace = curFace.astype(np.uint8)
  cv2.imwrite('fisherFace_' + str(count) + '.png', curFace)
  count += 1
'''
'''
pTest = (eigenVecs[:,:pNum*trainNum - pNum]).T @ (test - np.repeat(mean,pNum*testNum,axis = 1))

pTest = wVecs.T @ pTest
print(pTest.shape)

pTest = pTest[:projDim,:]
X = TSNE(n_components=2,perplexity=50).fit_transform(pTest.T)

plt.title("LDA scattering (DIM = 30)")
for i in range(pNum):
  plt.scatter(X[i*testNum:(i+1)*testNum,0],X[i*testNum:(i+1)*testNum,1])
plt.show()
'''