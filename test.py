#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import numpy as np
from skimage import transform, io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import core.module as module
import core.DataIter as DI
import core.config as config
import core.visualize as visualize



def heat_maps(weight, feature_maps, label):
    '''
    Compute heat map with the last convolutional layer feature maps and the
    weight matrix.
    params:
        weight: (M, C)
        feature_maps: (batch_size, M, h, w)
        label: (batch_size, )
    return:
        hms: (batch_size, )
    '''

    batch_size = feature_maps.shape[0]
    def compute_fms(product):
        psum = np.sum(product, axis=0)
        pmax, pmin = np.max(psum), np.min(psum)
        pnorm = (psum-pmin)/(pmax-pmin)
        pnorm = pnorm * 2 - 1
        pnorm_resize = transform.resize(pnorm,(32,32))
        return pnorm_resize

    hms = [compute_fms(weight[label[i],:].reshape(-1,1,1)*feature_maps[i,:,:,:]) for i in range(batch_size)]

    return hms


def img_save_to(heat_map, batch, prefix):
    def save_one(prefix, num, org, hm):
        name_org = ''.join([prefix, '-{}_org.jpg'.format(num)])
        name_hm = ''.join([prefix, '-{}_hm.jpg'.format(num)])
        io.imsave(name_org, org)
        io.imsave(name_hm, hm)

    batch_size = len(heat_map)
    [save_one(prefix, i, batch[i], heat_map[i]) for i in range(batch_size)]



def test(draw=False):
    # control parameters
    batch_num = config.generate_batches
    save_path = "./pictures_export/"

    # get trained module
    mod = module.get_test_module()

    # get dataIter
    _, it = DI.get_cifar10_iters()

    # Image Handler
    ig_org = visualize.ImgGrids(1)
    ig_hm = visualize.ImgGrids(2)


    # compute forward
    feature_maps = []
    weight = []
    count = 0
    for batch, label in it:
        if count > batch_num:
            break
        databatch = mx.io.DataBatch([batch.as_in_context(mx.gpu())], label=[label.as_in_context(mx.gpu())])
        mod.forward(databatch, is_train=False)
        out = mod.get_outputs()
        weight = out[1].asnumpy()
        feature_maps = out[2].asnumpy()

        # compute heat map
        hm = heat_maps(weight, feature_maps, label.asnumpy())

        # save heat_map and its associated image
        prefix = "".join([save_path, "batch{}".format(count)])
        org_batch = batch.asnumpy()
        img_save_to(hm, org_batch, prefix)

        count += 1

    if draw:
        ig_org.draw(org_batch.transpose(0,3,1,2)[0:64])
        ig_hm.draw(hm[:64])
        ch = input()


if __name__ == "__main__":
    test(True)

