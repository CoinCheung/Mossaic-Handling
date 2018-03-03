#!/usr/bin/python



import mxnet as mx
import numpy as np
import core.config as config


def trans(data, label):
    '''
    Since the CIFAR-10 dataset images have shapes of (32,32,3), they have to be
    transposed to 'NCWH' in order to be computed through the network
    '''
    data = data.astype(np.float32)
    mean = mx.nd.mean(data)
    data = mx.nd.transpose((data-mean), axes=(2,0,1))
    label.astype(np.uint8)
    return data, label


def get_cifar10_iters():
    cifar10_train = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=True,
        transform=trans
    )
    cifar10_test = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=False,
        transform=trans
    )

    train_data = mx.gluon.data.DataLoader(
        cifar10_train,
        config.batch_size,
        shuffle=True,
        last_batch ='rollover'
    )
    test_data = mx.gluon.data.DataLoader(
        cifar10_test,
        config.batch_size,
        shuffle=True,
        last_batch='rollover'
    )

    return train_data, test_data


