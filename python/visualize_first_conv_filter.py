#!/usr/bin/env python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
caffe_root = '/home/jiangliang/code/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('network', help = 'Network to use')
    parser.add_argument('model', help = 'Model to use')
    return parser

def vis_square(data):
    data = data[0 : 61, :]
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n * n - data.shape[0]), (0, 0), (1, 1), (1, 1))
    data = np.pad(data, padding, mode = 'constant', constant_values = 0)
    data = data.squeeze()
    total_array = np.array([])
    for i in range(n):
        row_array = np.array([])
        for j in range(n):
            if j == 0:
                row_array = data[i * n + j, :, :]
            else:
                row_array = np.hstack((row_array, data[i * n + j, :, :]))
        if i == 0:
            total_array = row_array
        else:
            total_array = np.vstack((total_array, row_array))
    print("shape of total_array: {}".format(total_array.shape))
    #data = data.reshape((n, n) + data.shape[1:])
    #data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(total_array);
    plt.axis('off')
    plt.show()

parser = argparser()
args = parser.parse_args()
network = args.network
model = args.model
net = caffe.Net(network, model, caffe.TEST)
filters = net.params['conv1_1'][0].data
vis_square(filters)

