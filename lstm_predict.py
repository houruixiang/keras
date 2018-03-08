'''
Trains a LSTM on the traces.
According to the paper Breaking Cryptographic Implementations Using Deep Learning Techniques
'''

import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import load_model

number_samples = 3253
number_traces = 10000
batch_size = 256
num_classes = 256
epochs = 2
max_features = 3200
index = np.arange(number_traces)
# np.random.shuffle(index)

traceset = np.load('convert0-10000.npz')
data = traceset['value'][index]
data = data.astype('float64')[:, :number_samples]
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
data_train = data[0:8000, :]
data_train = sequence.pad_sequences(data_train, maxlen=max_features)
data_train = data_train.reshape(8000, 40, 80)

x = data_train[0:1, :, :]

for pos in range(16):
    # define Long and Short Term Memory
    model = load_model('sca_model_SBox' + str(pos) + '.h5')
    label = traceset['label'][:, pos][index]
    cls = model.predict_classes(x, batch_size=32, verbose=0)
    print('Possible SBox' + str(pos) + ' Output:%02X,and label is %s' % (int(cls), label[0]))
