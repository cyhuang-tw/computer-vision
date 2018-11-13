import numpy as np
from numpy import linalg as LA

def PCA(data):
  num = data.shape[1]
  mean = np.mean(data,axis = 1,keepdims = True)

  cov = np.cov(data,bias=True)
  eigenVals, eigenVecs = LA.eig(cov)

  eigenVals = np.real(eigenVals[:num-1])
  eigenVecs = np.real(eigenVecs[:,:num-1])  
  '''
  idx = eigenVals.argsort()[::-1]   
  eigenVals = eigenVals[idx]
  eigenVecs = eigenVecs[:,idx]
  '''
  
  return eigenVals, eigenVecs, mean

def rebuildImg(img,eigenVecs,mean,nDims):
  img = img.flatten()
  img = img.reshape(img.shape[0],1)
  p = (eigenVecs.T) @ (img - mean)
  newImg = (eigenVecs[:,:nDims]) @ (p[:nDims,:]) + mean

  return newImg

def LDA(pData,pNum,trainNum):
  N = trainNum*pNum
  pVals = pData[:(N - pNum),:]
  Sw = np.zeros((N - pNum,N - pNum))
  Sb = np.zeros((N - pNum,N - pNum))
  means = np.zeros((N - pNum,pNum))

  for i in range(pNum):
    curMean = np.mean(pVals[:,i*trainNum:(i+1)*trainNum],axis = 1).reshape(N - pNum,1)
    for j in range(trainNum):
      curVec = pVals[:,i*trainNum + j].reshape(N-pNum,1)
      Sw = Sw + np.matmul((curVec - curMean),(curVec - curMean).T)
    means[:,i] = curMean.flatten()

  totalMean = np.mean(means,axis = 1,keepdims = True)
  for i in range(pNum):
    curMean = means[:,i].reshape(N-pNum,1)
    Sb = Sb + (curMean - totalMean) @ (curMean - totalMean).T

  wVals, wVecs = LA.eig(LA.inv(Sw) @ Sb)

  wVals = np.real(wVals[:pNum-1])
  wVecs = np.real(wVecs[:,:pNum-1])

  '''
  idx = wVals.argsort()[::-1]   
  wVals = wVals[idx]
  wVecs = wVecs[:,idx]
  '''

  return wVals, wVecs