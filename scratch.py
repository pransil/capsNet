""" scratch.py - a place to try small experiments, figure out how things work """

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import kernels

"""
class MultiLayerPerceptron(chainer.Chain):

    def __init__(self, n_in, n_hidden, n_out):
        super(MultilayerPerceptron, self).__init__()
        with self.init_scope():
            self.layer1 = L.Linear(n_in, n_hidden)
            self.layer2 = L.Linear(n_hidden, n_hidden)
            self.layer3 = L.Linear(n_hidden, n_out)

    def __call__(self, x):
        # Forward propagation
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        return self.layer3(h2)
"""
"""
class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))
"""
def main():

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    for i in range(6):
        l = train._datasets[1][0:10000][i]
        img = train._datasets[0][0:10000][i]
        print(l)

    n = 2
    c_i, c_o = 1, 512
    h_i, w_i = 28, 28
    h_k, w_k = 3, 3
    h_p, w_p = 1, 1
    z = train._datasets[0][0:n]
    z = z.reshape((n, 1, 28, 28))
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype('f')
    x.shape

    #kW = np.random.uniform(0, 1, (c_o, c_i, h_k, w_k)).astype('f')
    kW, k_props = kernels.make_kernels()
    kW.shape
    kS = kernels.make_similiarity_matrix(kW, k_props)
    csm = kernels.make_cross_support_matrix(len(kW))

    b = np.random.uniform(0, 1, (c_o,)).astype('f')
    b.shape

    s_y, s_x = 1, 1
    y = F.convolution_2d(x, kW, b, stride=(s_y, s_x), pad=(h_p, w_p))
    yz = F.convolution_2d(z, kW, b, stride=(s_y, s_x), pad=(h_p, w_p))
    y.shape
    kernels.update_csm(csm, yz)
    print(csm['p'])
    exit(-1)

    h_o = int((h_i + 2 * h_p - h_k) / s_y + 1)
    w_o = int((w_i + 2 * w_p - w_k) / s_x + 1)
    y.shape == (n, c_o, h_o, w_o)

    y = F.convolution_2d(x, kW, b, stride=(s_y, s_x), pad=(h_p, w_p), cover_all=True)
    y.shape == (n, c_o, h_o, w_o + 1)



def my_kernels():
    k = []
    k_props = np.zeros((512,2), dtype=int)
    k_props -= 1
    for i in range(512):
        a = kernels.make_array(i)
        k.append(a)
    for i in range(512):
        for j in range(i, 512):
            if i == j:
                continue
            if np.sum(k[j]) > np.sum(k[i]):
                continue
            if k_props[j][0] != -1:
                continue                   # Has already matched
            b = np.copy(k[j])
            r = kernels.rotation_check(k[i], b)
            if r >= 0:
                k_props[j] = [i, r]
                continue

    kk = np.reshape(k, (512,1,3,3))
    return kk



if __name__ == '__main__':
    main()
