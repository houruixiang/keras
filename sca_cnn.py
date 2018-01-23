'''Trains a simple convnet on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np

number_samples = 1000
number_train = 250000
number_test = 2500
batch_size = 128
num_classes = 256
epochs = 12

# Generate dummy data

data_train = np.random.random((number_train, 32, 32, 1))
data_test = np.random.random((number_test, 32, 32, 1))
print(data_train.shape[0], 'train samples')
print(data_test.shape[0], 'test samples')

label_train = keras.utils.to_categorical(np.random.randint(num_classes, size=(number_train, 1)), num_classes)
label_test = keras.utils.to_categorical(np.random.randint(num_classes, size=(number_test, 1)), num_classes)

# define Convolutionnal Neural Network
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), activation='tanh'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(data_train, label_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(data_test, label_test))
score = model.evaluate(data_test, label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
