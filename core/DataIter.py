#!/usr/bin/python



import mxnet as mx
import numpy as np
import core.config as config


def trans_train(data, label):
    '''
    Since the CIFAR-10 dataset images have shapes of (32,32,3), they have to be
    transposed to 'NCWH' in order to be computed through the network
    '''
    # transpose image data
    data = data.astype(np.float32)

    # image augmentation
    AugFun = [mx.img.ResizeAug(40), # resize to 40
              mx.img.RandomCropAug((32,32)), # crop 32x32 sum-images
              mx.img.HorizontalFlipAug(.5), # flip randomly
              ]
    for fun in AugFun:
        data = fun(data)

    label.astype(np.uint8)
    return data, label


def trans_test(data, label):
    '''
    Since the CIFAR-10 dataset images have shapes of (32,32,3), they have to be
    transposed to 'NCWH' in order to be computed through the network
    '''
    #  data = mx.nd.transpose(data, axes=(2,0,1))
    label.astype(np.uint8)
    return data, label



def get_cifar10_iters():
    batch_size = config.batch_size

    cifar10_train = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=True,
        transform=trans_train
    )
    cifar10_test = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=False,
        transform=trans_test
    )

    train_data = mx.gluon.data.DataLoader(
        cifar10_train,
        batch_size,
        shuffle = True,
        last_batch ='rollover',
        #  num_workers = 4
    )
    test_data = mx.gluon.data.DataLoader(
        cifar10_test,
        batch_size,
        shuffle = True,
        last_batch ='rollover',
        #  num_workers = 4
    )

    return train_data, test_data

