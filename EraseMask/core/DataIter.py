#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import numpy as np
import os
import random
import core.config as config


batch_size = config.batch_size

def get_dataiter(batch_size = batch_size):
    home_dir = os.path.expandvars('$HOME')
    train_path = home_dir + '/.mxnet/datasets/MaskDataSet/Erase/train.rec'
    val_path = home_dir + '/.mxnet/datasets/MaskDataSet/Erase/val.rec'
    seed = random.randint(0, 5000)

    img_shape = (3, 224, 448)
    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_path,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
    )
    val_iter = mx.io.ImageRecordIter(
        path_imgrec=val_path,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
    )

    return train_iter, val_iter


# normalize an image to range [-1, 1]
def img_norm(img):
    img /= 127.5
    img -= 1
    return img


# opposite operation of img_norm() which convert an image from [-1, 1] to [0, 255]
def img_recover(img):
    img += 1
    img *= 127.5
    img = img.astype(np.uint8)
    return img
