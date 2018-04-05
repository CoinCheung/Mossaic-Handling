#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import os
import numpy as np
from skimage import transform, io
import core.config as config
from core.DataIter import get_dataiter, img_norm, img_recover
from collections import namedtuple
from core.Network import get_test_network


pics_export_path = os.getcwd() + '/pics_export/'
batch_size = config.test_batch_size
gen_batch_num = config.generate_batch_num



def img_save_one(img, prefix):
    img = img.asnumpy().transpose(1,2,0)
    io.imsave(prefix, img)


def test():
    if not os.path.exists(pics_export_path):
        os.mkdir(pics_export_path)

    net = get_test_network()

    train_it, val_it = get_dataiter(batch_size)
    Batch = namedtuple('batch', ['data'])

    num = 0
    for i, batch in enumerate(val_it):
        data_org, data_masked = mx.nd.split(batch.data[0], axis=3, num_outputs=2)

        data_masked = img_norm(data_masked).as_in_context(mx.gpu())
        net.forward(Batch([data_masked]))
        out = net.get_outputs()[0]

        ## export corresponding images
        prefix = pics_export_path + 'batch'
        [img_save_one(img_recover(out[n]), prefix + '{}_{}_erased.jpg'.format(i, n))
                for n in range(batch_size)]
        [img_save_one(img_recover(data_masked[n]), prefix + '{}_{}_masked.jpg'.format(i, n))
                for n in range(batch_size)]
        [img_save_one(data_org[n].astype(np.uint8), prefix + '{}_{}_org.jpg'.format(i, n))
                for n in range(batch_size)]

        num += 1
        if num == gen_batch_num:
            break


if __name__ == "__main__":
    test()
