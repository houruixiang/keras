'''
Trains a LSTM on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt

number_samples = 3253
number_traces = 10000
batch_size = 16
num_classes = 9
epochs = 10
max_features = 3200
index = np.arange(number_traces)
np.random.shuffle(index)


def showplt(y0, y1):
    x = np.arange(0, number_samples, 1)
    plt.figure(figsize=(100, 10))
    plt.plot(x, y0, 'r', x, y1, 'b')
    plt.savefig('result.png')
    plt.show()


traceset = np.load('convert0-10000.npz')
data = traceset['value'][index]
label = traceset['HW'][index]
data = data.astype('float64')
data_train = data[0:8000, :]
data_train -= np.mean(data, axis=0)
data_train /= np.std(data, axis=0)
label_train = keras.utils.to_categorical(label[0:8000], num_classes=9)
data_test = data[8000:, :]
data_test -= np.mean(data, axis=0)
data_test /= np.std(data, axis=0)
label_test = keras.utils.to_categorical(label[8000:], num_classes=9)

data_train = sequence.pad_sequences(data_train, maxlen=max_features)
data_test = sequence.pad_sequences(data_test, maxlen=max_features)
data_train = data_train.reshape(8000, 40, 80)
data_test = data_test.reshape(2000, 40, 80)

# define Long and Short Term Memory
model = Sequential()
model.add(LSTM(26, return_sequences=True, input_shape=(40, 80)))
model.add(LSTM(26))
# model.add(LSTM(26))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs)
score = model.evaluate(data_test, label_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
