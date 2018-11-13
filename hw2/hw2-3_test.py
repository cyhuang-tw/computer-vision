import os
import sys
import numpy as np
import cv2
import csv
import keras
from keras.models import load_model

model = load_model('model.h5')

img_size = 28

path = sys.argv[1]
outPath = sys.argv[2]

fileList = os.listdir(path)

data = np.zeros((len(fileList),img_size,img_size))

i = 0
for file in fileList:
  img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
  data[i,:,:] = img
  i += 1

data = data / 255

data = data.reshape(data.shape + (1,))
ans = model.predict(data)

csvFile = open(outPath,'w',newline='')

writer = csv.writer(csvFile)
writer.writerow(['id','label'])

idx = 0
for file in fileList:
  label = ans[idx,:]
  num = np.argmax(label)
  fileName = os.path.splitext(file)[0]
  writer.writerow([fileName,num])
  idx += 1