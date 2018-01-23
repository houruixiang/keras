'''Trains a LSTM on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

import keras
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

max_features = 1001

# define Long and Short Term Memory
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(26))
model.add(LSTM(26))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(data_train, label_train, batch_size=16, epochs=10)
score = model.evaluate(data_test, label_test, batch_size=16)
