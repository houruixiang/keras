'''
Trains a LSTM on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras, math
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import batch_read_data

# import wechat_utils

number_samples = 225
number_traces = 10000
number_train = int(number_traces * 0.8)
number_test = int(number_traces * 0.2)
batch_size = 256
num_classes = 256
epochs = 40
shape = (40, 80)
max_features = shape[0] * shape[1]

index = np.arange(number_traces)
np.random.shuffle(index)

traceset = np.load('convert0-10000.npz')
data = traceset['value'][index]
data = data.astype('float64')[:, :number_samples]
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

data_train = data[0:number_train, :]
data_test = data[number_train:number_traces, :]
data_train = sequence.pad_sequences(data_train, maxlen=max_features)
data_test = sequence.pad_sequences(data_test, maxlen=max_features)
data_train = data_train.reshape(number_train, shape[0], shape[1])
data_test = data_test.reshape(number_test, shape[0], shape[1])

# define Long and Short Term Memory
model = Sequential()
model.add(LSTM(26, return_sequences=True, input_shape=shape))
model.add(LSTM(26))
# model.add(LSTM(26))
model.add(Dense(num_classes, activation='softmax'))
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
x = data_train[0:1, :, :]
for pos in range(1):
    label = traceset['label'][:, pos][index]
    for i in range(len(label)):
        label[i] = int(label[i], 16)
    label_train = keras.utils.to_categorical(label[0:number_train], num_classes)
    label_test = keras.utils.to_categorical(label[number_train:], num_classes)
    model.fit_generator(
        batch_read_data.generate_arrays_from_file_batch('data', shape, 0, batch_size, 100, num_classes),
        verbose=1,
        epochs=epochs, steps_per_epoch=100)
    model.save('temp.h5')
    score = model.evaluate(data_test, label_test, batch_size=batch_size, verbose=1)
    model.save('sca_model_SBox' + str(pos) + '.h5')
    print('sca_model_SBox' + str(pos) + ' Test loss:', score[0])
    print('sca_model_SBox' + str(pos) + ' Test accuracy:', score[1])
    cls = model.predict_classes(x, batch_size=32, verbose=0)
    print('Possible SBox' + str(pos) + ' Output:%02X,and label is %s' % (int(cls), label[0]))
