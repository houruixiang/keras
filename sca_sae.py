'''
Trains a stacked auto-encoder on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import batch_read_data

# import wechat_utils
for number_samples in range(3253, 3254, 100):
    print(number_samples)
    start = 100
    end = start + number_samples
    number_traces = 10000
    batch_size = 16
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

    # define Stacked Auto-Encoder, attention:the paper did not mention if there is a dropout layer
    model = Sequential()
    # input layer,depends on samples per trace
    model.add(Dense(number_samples, activation='relu', input_shape=(number_samples,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,  # we can also use optimizer sgd for a possible better result
                  metrics=['accuracy'])

    model.fit_generator(batch_read_data.generate_arrays_from_file('./data', 0, 10000, 128), verbose=2,
                        samples_per_epoch=number_traces,
                        epochs=epochs)
    score = model.evaluate(data_test, label_test, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
