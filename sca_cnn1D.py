'''
Trains a simple convnet on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD

# import wechat_utils


for number_samples in range(300, 500, 100):
    print(number_samples)
    start = 0
    end = start + number_samples
    number_traces = 10000
    batch_size = 256
    num_classes = 256
    epochs = 100
    index = np.arange(number_traces)
    np.random.shuffle(index)

    traceset = np.load('convert0-10000.npz')
    data = traceset['value'][:, start:end][index]
    label = traceset['label'][:, 0][index]
    data = data.astype('float64')
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    for i in range(len(label)):
        label[i] = int(label[i], 16)
    data_train = data[0:6000, :]
    data_val = data[6000:8000, :]
    data_test = data[8000:number_traces, :]
    label_train = keras.utils.to_categorical(label[0:6000], num_classes=num_classes)
    label_val = keras.utils.to_categorical(label[6000:8000], num_classes=num_classes)
    label_test = keras.utils.to_categorical(label[8000:number_traces], num_classes=num_classes)

    # define Convolutional Neural Network
    model = Sequential()
    model.add(Reshape((number_samples, 1), input_shape=(number_samples,)))  # reshape data for convolutional layer
    model.add(Conv1D(8, kernel_size=16, padding='valid', activation='relu'))
    model.add(Dropout(0.8))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    model.add(Conv1D(8, kernel_size=8, padding='valid', activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Flatten())

    model.add(Dense(num_classes))

    model.summary()
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(data_train, label_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(data_val, label_val))
    score = model.evaluate(data_test, label_test, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
