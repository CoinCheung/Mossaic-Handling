#!/usr/bin/python


import mxnet as mx



## symbols


def generator():
    slope = 0.2
    filter_base = 64

    img = mx.sym.var('img')
    ### unet
    ## encoding
    # 3x256x256
    conv = mx.sym.Convolution(img, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv1')
    relu = mx.sym.LeakyRelu(conv, act_type='leaky', slope=slope, name='gen_leaky1')
    # 64x128x128
    for i in range(2, 6):
        filter_base *= 2
        conv = mx.sym.Convolution(relu, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv'+str(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='gen_down_bn'+str(i))
        relu = mx.sym.LeakyRelu(bn, act_type='leaky', slope=slope, name='gen_leaky'+str(i))
    # 1024x8x8
    for i in range(1, 5):
        filter_base /= 2
        dconv = mx.sym.Deconvolution(relu, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=filter_base, name='gen_dconv'+str(i))
        bn = mx.sym.BatchNorm(dconv, fix_gamma=False, name='gen_up_bn'+str(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='gen_relu'+str(i))
    # 64x128x128
    dconv = mx.sym.Deconvolution(relu, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, name='dconv_last')
    img_gen = mx.sym.Activation(dconv, act_type='tanh', name='img_gen')
    # 3x256x256

    return img_gen


def discriminator():
    imgA = mx.sym.var('out_img')
    imgB = mx.sym.var('original_img')
    label = mx.sym.var('label')
    img = mx.sym.concat(imgA, imgB, dim=1)

