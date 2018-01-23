import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class trace(object):
    def __init__(self, load_fn):
        self.load_fn = load_fn
        load_data = sio.loadmat(load_fn)
        self.number_samples = len(load_data['sampleBlock0'][0])
        num_block = int((len(load_data) - 10) / 2)
        sampleset = []
        dataset = []
        for i_block in range(num_block):
            name_sample_block = 'sampleBlock' + str(i_block)
            name_data_block = 'dataBlock' + str(i_block)
            sample_block = load_data[name_sample_block]
            data_block = load_data[name_data_block]
            for i_sample_block in range(len(sample_block)):
                sample_value = sample_block[i_sample_block]
                data_value = data_block[i_sample_block]
                sampleset.append(sample_value)
                dataset.append(data_value)
        self.traceset = dict({'data': dataset, 'sample': sampleset})

    def savetomat(self, save_fn):
        sio.savemat(save_fn, self.traceset)

    def showplt(self, label, y):
        x = np.arange(0, self.number_samples, 1)
        plt.figure(figsize=(100, 5))
        plt.xlabel(label)
        plt.plot(x, y)
        plt.show()


def int8tohex(num_array):
    hex_array = []
    for num in num_array:
        a = hex(num + 128)
        hex_array.append(a)
    return hex_array


if __name__ == "__main__":
    # a = trace('98.mat')
    a = trace('AES_without_countmeasure.mat')
    traceset = a.traceset
    for i in range(1):
        trace = traceset['sample']
        label = traceset['data'][0][:16]
        a.showplt(int8tohex(label), trace[i])
    # a.savetomat('save.mat')
