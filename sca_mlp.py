'''
Trains a multi-layer perceptron on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
import wechat_utils

number_samples = 3253
number_traces = 10000
batch_size = 128
num_classes = 256
epochs = 12
index = np.arange(number_traces)
np.random.shuffle(index)

traceset = np.load('convert0-10000.npz')
data = traceset['value'][index]
label = traceset['label'][index]
data = data.astype('float64')[:, :number_samples]
data_train = data[0:8000, :]
data_train -= np.mean(data, axis=0)
data_train /= np.std(data, axis=0)
label_train = keras.utils.to_categorical(label[0:8000], num_classes=num_classes)
data_test = data[8000:number_traces, :]
data_test -= np.mean(data, axis=0)
data_test /= np.std(data, axis=0)
label_test = keras.utils.to_categorical(label[8000:], num_classes=num_classes)

# define Multilayer Perceptron, attention:the paper did not mention if there is a dropout layer
model = Sequential()
# input layer,depends on samples per trace
model.add(Dense(number_samples, activation='relu', input_shape=(number_samples,)))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),  # we can also use optimizer sgd for a possible better result
              metrics=['accuracy'])

model.fit(data_train, label_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(data_test, label_test),callbacks=[wechat_utils.sendmessage(savelog=True, fexten='TEST')])
score = model.evaluate(data_test, label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
