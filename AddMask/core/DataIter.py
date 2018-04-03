#!/usr/bin/python



import mxnet as mx
import numpy as np
import random
import os
import core.config as config


def trans_train_cifar10(data, label):
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


def trans_test_cifar10(data, label):
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
        transform=trans_train_cifar10
    )
    cifar10_test = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=False,
        transform=trans_test_cifar10
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


def get_selfmade_iters():
    home_dir = os.path.expandvars('$HOME')
    batch_size = config.batch_size
    train_pth = home_dir + '/.mxnet/datasets/MaskDataSet/Add/train.rec'
    val_pth = home_dir + '/.mxnet/datasets/MaskDataSet/Add/val.rec'
    img_shape = (3, 224, 224)
    seed = random.randint(0, 5000)

    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_pth,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
        rand_mirror=True,
        resize=300,
    )
    val_iter = mx.io.ImageRecordIter(
        path_imgrec=val_pth,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
        rand_mirror=True,
        resize=300,
    )

    return train_iter, val_iter
