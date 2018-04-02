#!/usr/bin/python
# -*- encoding: utf8 -*-

import mxnet as mx
import numpy as np
import os
from skimage import transform, io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import core.module as module
import core.DataIter as DI
import core.config as config
import core.visualize as visualize
import core.clibs as clibs



image_size = config.image_size
mossaic_ratio = config.mossaic_ratio
save_path = config.save_path


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
        pnorm_resize = transform.resize(pnorm,(image_size,image_size))
        return pnorm_resize
    label = [int(el) for el in label]

    hms = [compute_fms(weight[label[i],:].reshape(-1,1,1)*feature_maps[i,:,:,:]) for i in range(batch_size)]

    return hms


def img_save_to(heat_map, batch, prefix):
    def save_one(prefix, num, org, hm):
        name_org = ''.join([prefix, '-{}_org.jpg'.format(num)])
        name_hm = ''.join([prefix, '-{}_hm.jpg'.format(num)])
        name_mask = ''.join([prefix, '-{}_mossaic.jpg'.format(num)])
        io.imsave(name_org, org)
        io.imsave(name_hm, hm)
        return (name_org, name_hm, name_mask)

    batch_size = len(heat_map)
    name_tuples = [save_one(prefix, i, batch[i], heat_map[i]) for i in
            range(batch_size)]

    return name_tuples


def test_resnet_cifar10(draw=False):
    # control parameters
    batch_num = config.generate_batches

    # get trained module
    mod = module.get_test_module_resnet_cifar10()

    # get dataIter
    _, it = DI.get_cifar10_iters()

    # Image Handler
    ig_org = visualize.ImgGrids(4)
    ig_hm = visualize.ImgGrids(5)

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
        ch = input('Input something to end the program: ')



def test_resnet_imagenet():
    # control parameters
    batch_num = config.generate_batches

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get trained module
    mod = module.get_test_module_resnet_imagenet()

    # get dataIter
    _, it = DI.get_selfmade_iters()

    # compute forward
    feature_maps = []
    weight = []
    path_tuples = []
    count = 0
    for batch in it:
        if count > batch_num:
            break
        mod.forward(batch, is_train=False)
        out = mod.get_outputs()
        weight = out[1].asnumpy()
        feature_maps = out[2].asnumpy()
        org_batch = batch.data[0].asnumpy()

        label = batch.label[0]

        # compute heat map
        hm = heat_maps(weight, feature_maps, label.asnumpy())

        # save heat_map and its associated image
        prefix = "".join([save_path, "batch{}".format(count)])
        org_batch = [el.transpose(1,2,0).astype(np.uint8) for el in org_batch]
        one_path_tuple = img_save_to(hm, org_batch, prefix)
        path_tuples.extend(one_path_tuple)

        count += 1

    # add mossaic according to the heat maps and save the masked images
    fun = clibs.add_mossaic_fun()
    [fun(el[0].encode('utf-8'), el[1].encode('utf-8'), el[2].encode('utf-8'),
        mossaic_ratio) for el in path_tuples]

    if config.gen_heatmap == False:
        [os.remove(el[1]) for el in path_tuples]



def gen_mossaic_training_dataset():
    # control parameters
    batch_num = config.generate_batches
    save_path = os.getcwd() + "/../dataset/Mossaic_JPEG/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get trained module
    mod = module.get_test_module_resnet_imagenet()

    # get dataIter
    it, _ = DI.get_selfmade_iters()

    print('===========================')
    print('compute the heat maps')
    print('===========================')
    # compute forward
    feature_maps = []
    weight = []
    path_tuples = []
    count = 0
    for batch in it:
        mod.forward(batch, is_train=False)
        out = mod.get_outputs()
        weight = out[1].asnumpy()
        feature_maps = out[2].asnumpy()
        org_batch = batch.data[0].asnumpy()

        label = batch.label[0]

        # compute heat map
        hm = heat_maps(weight, feature_maps, label.asnumpy())

        # save heat_map and its associated image
        prefix = "".join([save_path, "batch{}".format(count)])
        org_batch = [el.transpose(1,2,0).astype(np.uint8) for el in org_batch]
        one_path_tuple = img_save_to(hm, org_batch, prefix)
        path_tuples.extend(one_path_tuple)

        count += 1


    print('===========================')
    print('start to generate dataset')
    print('===========================')
    # write the original image and heat map image pathes to a file
    namefile = save_path+"/.filenames.txt"
    print(save_path)
    with open(namefile, 'w') as wf:
        [wf.write('{}\n{}\n'.format(el[0], el[1])) for el in path_tuples]

    # get c++ function
    fun = clibs.gen_mossaic_dataset()
    fun(namefile.encode('utf_8'), mossaic_ratio)

    [os.remove(el[1]) for el in path_tuples]
    [os.remove(el[0]) for el in path_tuples]
    os.remove(namefile)


if __name__ == "__main__":
    #  test_resnet_cifar10(True)
    test_resnet_imagenet()
    #  gen_mossaic_training_dataset()
