"""
using fit_generator to train a model with batch_size(e.g. 128) of samples in a cycle
"""
import os, keras
import numpy as np


def generate_arrays_from_file_batch(dir, shape, batch_size=256, num_per_label=100, num_classes=1):
    """
    training samples batch by batch
    :param dir: set the dir of samples
    :param start: set start point of a sample
    :param end: set end point of a sample
    :param batch_size: numbers of a batch
    :param shape: sometimes we want to reshape the samples to 2D or 3D
    :param num_classes: use in categorical classification
    :return:a numpy array formats like (samples,labels)
    """

    while 1:
        sampleset = ArrangeFileName('dataset', num_per_label, num_classes)
        cnt = 0
        X = []
        Y = []
        for sample in sampleset:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_file(dir, sample[0].strip('\n'), sample[1])
            if -1 != shape[0]:
                X.append(x[:shape[0] * shape[1]])
            else:
                X.append(x)
            Y.append(y)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                np_X = np.array(X).astype('float64')
                if -1 != shape[0]:
                    np_X = np_X.reshape(batch_size, shape[0], shape[1])
                np_Y = keras.utils.to_categorical(np.array(Y), num_classes)
                np_X -= np.mean(np_X, axis=0)
                np_X /= np.std(np_X, axis=0)
                yield (np_X, np_Y)
                X = []
                Y = []
    f.close()


def process_file(dir, filename, labelname):
    value = getvalue(os.path.join(dir, filename))
    label = int(labelname, 16)
    return (value, label)


def ArrangeFileName(dir, num_per_label, num_classes):
    cnt = 0
    samplefileset = []
    sampleset = [['' for col in range(2)] for row in range(num_per_label * num_classes)]
    for file in os.listdir(dir):
        filename = os.path.join(dir, file)
        if os.path.isfile(filename) and os.path.splitext(filename)[1] == ".txt":
            samplefileset.append(filename)
    for samplefile in samplefileset:
        cls = os.path.splitext(os.path.split(samplefile)[1])[0]
        with open(samplefile, 'r') as g:
            samples = g.readlines()
            for i in range(num_per_label):
                sampleset[cnt + num_classes * i] = samples[i], cls
        cnt += 1
    return sampleset


def getvalue(csvname):
    value = []
    file = open(csvname)
    for line in file:
        value.append(line)
    return value[24:]
