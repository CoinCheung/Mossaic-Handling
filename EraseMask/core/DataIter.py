#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import os
import random
import core.config as config


batch_size = config.batch_size

def get_dataiter():
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
    max_ = mx.nd.max(img)
    min_ = mx.nd.min(img)
    img = (2 * img - max_ - min_) / (max_ - min_)
    return img

