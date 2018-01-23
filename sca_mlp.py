'''Trains a simple deep NN on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD

import numpy as np

number_samples = 1000
number_train = 250000
number_test = 2500
batch_size = 128
num_classes = 10
epochs = 20

# Generate dummy data

data_train = np.random.random((number_train, number_samples))
data_test = np.random.random((number_test, number_samples))
print(data_train.shape[0], 'train samples')
print(data_test.shape[0], 'test samples')

label_train = keras.utils.to_categorical(np.random.randint(num_classes, size=(number_train, 1)), num_classes)
label_test = keras.utils.to_categorical(np.random.randint(num_classes, size=(number_test, 1)), num_classes)

# define Multilayer Perceptron, attention:the paper did not mention if there is a dropout layer
model = Sequential()
# input layer,depends on samples per trace
model.add(Dense(number_samples, activation='relu', input_shape=(number_samples,)))
# model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),  # we can also use optimizer sgd for a possible better result
              metrics=['accuracy'])

history = model.fit(data_train, label_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(data_test, label_test))
score = model.evaluate(data_test, label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
