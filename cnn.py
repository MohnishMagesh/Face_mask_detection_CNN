import numpy as np

data = np.load('data.npy')
labels = np.load('labels.npy')

from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

# model.add(conv2D,(200,(3,3),))