import numpy as np

data = np.load('data.npy')
labels = np.load('labels.npy')

from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout,Conv2D,MaxPooling2D
# from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# training the models
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

print(model.evaluate(test_data, test_labels))