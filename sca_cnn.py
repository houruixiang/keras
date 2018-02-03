'''
Trains a simple convnet on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D

number_samples = 3249
number_traces = 10000
batch_size = 128
num_classes = 9
kernel_size = (3, 3)
epochs = 12
index = np.arange(number_traces)
np.random.shuffle(index)

traceset = np.load('convert0-10000.npz')
data = traceset['value'][index]
label = traceset['HW'][index]
data = data.astype('float64')[:, :number_samples]
data_train = data[0:8000, :]
data_train -= np.mean(data, axis=0)
data_train /= np.std(data, axis=0)
label_train = keras.utils.to_categorical(label[0:8000], num_classes=num_classes)
data_test = data[8000:, :]
data_test -= np.mean(data, axis=0)
data_test /= np.std(data, axis=0)
label_test = keras.utils.to_categorical(label[8000:], num_classes=num_classes)

# define Convolutional Neural Network
model = Sequential()
model.add(Reshape((57, 57, 1), input_shape=(number_samples,)))  # reshape data for convolutional layer
model.add(Conv2D(8, kernel_size=kernel_size, padding='valid', activation='relu', input_shape=(57, 57, 1)))
model.add(Conv2D(8, kernel_size, activation='relu'))
model.add(Dropout(0.75))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size, activation='relu'))
model.add(Dropout(0.75))
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
