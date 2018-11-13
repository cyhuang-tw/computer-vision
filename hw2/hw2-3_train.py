import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

path = sys.argv[1]

trainPath = os.path.join(path,'train')
valPath = os.path.join(path,'valid')
img_size = 28

trainData = np.ndarray((0,img_size,img_size))
trainLabel = np.ndarray(0)
valData = np.ndarray((0,img_size,img_size))
valLabel = np.ndarray(0)

trainFolder = os.listdir(trainPath)
valFolder = os.listdir(valPath)

for folder in trainFolder:
	label = int((folder.split('_'))[1])
	files = os.listdir(os.path.join(trainPath,folder))
	trainLabel = np.append(trainLabel,label*np.ones((1,len(files))))
	tmpData = np.empty((len(files),img_size,img_size))
	i = 0
	for file in files:
		img = cv2.imread(os.path.join(trainPath,folder,file),cv2.IMREAD_GRAYSCALE).reshape(1,img_size,img_size)
		tmpData[i,:,:] = img
		i += 1
	trainData = np.append(trainData,tmpData,axis = 0)
for folder in valFolder:
	label = int((folder.split('_'))[1])
	files = os.listdir(os.path.join(valPath,folder))
	valLabel = np.append(valLabel,label*np.ones((1,len(files))))
	tmpData = np.empty((len(files),img_size,img_size))
	i = 0
	for file in files:
		img = cv2.imread(os.path.join(valPath,folder,file),cv2.IMREAD_GRAYSCALE).reshape(1,img_size,img_size)
		tmpData[i:,:] = img
		i += 1
	valData = np.append(valData,tmpData,axis = 0)

trainData = trainData.reshape(trainData.shape + (1,))
valData = valData.reshape(valData.shape + (1,))
#print(trainData.shape)

trainData = trainData / 255
valData = valData / 255

trainLabel = keras.utils.to_categorical(trainLabel, 10)
valLabel = keras.utils.to_categorical(valLabel, 10)
#print(trainLabel.shape)

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
hist = model.fit(trainData, trainLabel, batch_size=16, epochs=20, verbose=1, validation_data=(valData, valLabel))
score = model.evaluate(valData, valLabel)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
#model.summary()
'''
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
'''

model.save('model.h5')