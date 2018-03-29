#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config
import symbol.meric as meric


def pre_processing(data, label):
    batch_size = config.batch_size

    # transpose and minus average
    #  data = mx.sym.transpose(data, axes=(0,1,2,3))
    #  data = mx.sym.transpose(data, axes=(0,3,1,2))
    data = mx.sym.broadcast_sub(data, mx.sym.reshape(mx.sym.mean(data, axis=(1,2,3)), shape=(batch_size,1,1,1)))

    return data, label



def SoftmaxLoss_with_Acc(scores, label):
    # used control parameters
    cls_num = config.cls_num
    # softmax loss
    score_max = mx.sym.max(scores, axis=1)
    score_clean = mx.sym.broadcast_sub(scores, score_max.reshape(shape=(-1,1)))
    score_clean_softmax_log = mx.sym.log_softmax(score_clean, axis=1)

    label_one_hot = mx.sym.one_hot(label, cls_num)

    product_sum = mx.sym.sum(label_one_hot*score_clean_softmax_log, axis=1)
    loss = -mx.sym.mean(product_sum)

    # predicted class
    pred = mx.sym.argmax(scores, axis=1)
    acc = meric.accuracy(pred, label)

    return loss, acc




def resnet_cifar10(n, cls_num):
    '''
        This is the resent-18 used for cifar-10 dataset. The structure is the
        same as that used in the experiment in the paper. By the way, in the paper,
        the network structure used for cifar-10 is different from imagenet dataset
        params:
            - n: the parameter n in the paper. With n = {3, 5, 7, 9}, the
            network is led to 20, 32, 44 layers.
            - cls_num: the number of classes for classification. For cifar-10,
            it is 10.
    '''
    img = mx.sym.var("img")
    label = mx.sym.var("label")

    # 3x32x32
    conv32 = mx.sym.Convolution(img, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv1')
    conv32 = mx.sym.Dropout(conv32, 0.2, 'training')
    # 16x32x32
    for i in range(layer_num):
        conv = mx.sym.Convolution(conv32, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn2_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu2_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv3_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn3_{}'.format(i))
        conv32 = bn + conv32
        conv32 = mx.sym.Activation(conv32, act_type='relu', name='relu3_{}'.format(i))

    # 16x32x32
    conv = mx.sym.Convolution(conv32, num_filter=32, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv4_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn4_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu4_0')
    conv = mx.sym.Convolution(relu, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn5_0')
    shortcut = mx.sym.Convolution(conv32, num_filter=32, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut0')
    shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, name='bn5_sc0')
    conv16 = bn + shortcut
    conv16 = mx.sym.Activation(conv16, act_type='relu', name='relu5_0')

    # 32x16x16
    for i in range(1,layer_num):
        conv = mx.sym.Convolution(conv16, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv4_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn4_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu4_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn5_{}'.format(i))
        conv16 = bn + conv16
        conv16 = mx.sym.Activation(conv16, act_type='relu', name='relu5_{}'.format(i))

    # 32x16x16
    conv = mx.sym.Convolution(conv16, num_filter=64, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv6_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn6_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu6_0')
    conv = mx.sym.Convolution(relu, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv7_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn7_0')
    shortcut = mx.sym.Convolution(conv16, num_filter=64, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut1')
    shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, name='bn7_sc0')
    conv8 = bn + shortcut
    conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu7_0')

    # 64x8x8
    for i in range(1,layer_num):
        conv = mx.sym.Convolution(conv8, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv6_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn6_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu6_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv7_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn7_{}'.format(i))
        conv8 = bn + conv8
        conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu7_{}'.format(i))

    # avg pooling, dense
    avg_pool = mx.sym.Pooling(conv8, global_pool=True, kernel=(8,8), pool_type='avg')
    # 64x1x1
    weight = mx.sym.var('weight')
    flatten = mx.sym.flatten(avg_pool)
    scores = mx.sym.FullyConnected(flatten, weight=weight, num_hidden=cls_num, no_bias=True)

    # loss output for training
    softmax_output = mx.sym.SoftmaxOutput(scores, label=label)
    # loss
    loss, acc = SoftmaxLoss_with_Acc(scores, label)

    # return value
    weight_out = mx.sym.BlockGrad(weight)
    conv8_out = mx.sym.BlockGrad(conv8)
    loss_out = mx.sym.BlockGrad(loss)
    acc_out = mx.sym.BlockGrad(acc)
    out = mx.sym.Group([softmax_output, weight_out, conv8_out, loss_out, acc_out])
    #  out = mx.sym.Group([softmax_output, scores_out, weight_out])

    return out



def resnet_imagenet(layer_num, cls_num):
    '''
    This defines the network structure used for imagenet in the paper without
    bottleneck structure. Thus it only supports 18-layer and 34-layer resnet.
    params:
        - layer_num: the number of layers. An arg layer_num of {18, 34} means the
        number of {conv2_x, conv3_x, conv4_x, conv5_x} to be {{2, 2, 2, 2}, {3, 4, 6, 3}}
        - cls_num: the number of output class numbers
    '''
    img = mx.sym.var("img")
    label = mx.sym.var("label")

    if layer_num == 18:
        units_num = [2,2,2,2]
    elif layer_num == 34:
        units_num = [3,4,6,3]


    #  debug_out = img
    # implement pre-processing
    img, label =  pre_processing(img, label)

    # 3x224x224
    conv = mx.sym.Convolution(img, num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3), no_bias=False, name='conv1')
    # 64x112x112
    pool = mx.sym.Pooling(conv, kernel=(3,3), stride=(2,2), pool_type='max', pad=(1,1), name='max_pool')

    #  conv32 = mx.sym.Dropout(conv32, 0.2, 'training')
    # 64x56x56
    conv_in = pool
    for i in range(units_num[0]):
        conv = mx.sym.Convolution(conv_in, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv21_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn21_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu21_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv22_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn22_{}'.format(i))
        conv_in = bn + conv_in
        conv_in = mx.sym.Activation(conv_in, act_type='relu', name='relu22_{}'.format(i))

    # 64x56x56
    conv = mx.sym.Convolution(conv_in, num_filter=128, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv31_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn31_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu31_0')
    # 128x28x28
    conv = mx.sym.Convolution(relu, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv32_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn32_0')
    shortcut = mx.sym.Convolution(conv_in, num_filter=128, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut0')
    shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, name='bn3_sc0')
    bn_sum = bn + shortcut
    relu = mx.sym.Activation(bn_sum, act_type='relu', name='relu32_0')

    # 128x28x28
    conv_in = relu
    for i in range(1,units_num[1]):
        conv = mx.sym.Convolution(conv_in, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv31_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn31_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu31_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv32_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn32_{}'.format(i))
        conv_in = bn + conv_in
        conv_in = mx.sym.Activation(conv_in, act_type='relu', name='relu32_{}'.format(i))

    # 128x28x28
    conv = mx.sym.Convolution(conv_in, num_filter=256, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv41_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn41_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu41_0')
    # 256x14x14
    conv = mx.sym.Convolution(relu, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv42_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn42_0')
    shortcut = mx.sym.Convolution(conv_in, num_filter=256, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut1')
    shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, name='bn42_sc0')
    bn_sum = bn + shortcut
    relu = mx.sym.Activation(bn_sum, act_type='relu', name='relu42_0')

    # 256x14x14
    conv_in = relu
    for i in range(1,units_num[2]):
        conv = mx.sym.Convolution(conv_in, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv41_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn41_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu41_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv42_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn42_{}'.format(i))
        conv_in = bn + conv_in
        conv_in = mx.sym.Activation(conv_in, act_type='relu', name='relu42_{}'.format(i))

    # 256x14x14
    conv = mx.sym.Convolution(conv_in, num_filter=512, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv51_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn51_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu51_0')
    # 512x7x7
    conv = mx.sym.Convolution(relu, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv52_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn52_0')
    shortcut = mx.sym.Convolution(conv_in, num_filter=512, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut2')
    shortcut = mx.sym.BatchNorm(shortcut, fix_gamma=False, name='bn52_sc0')
    bn_sum = bn + shortcut
    relu = mx.sym.Activation(bn_sum, act_type='relu', name='relu52_0')

    # 512x7x7
    conv_in = relu
    for i in range(1,units_num[3]):
        conv = mx.sym.Convolution(conv_in, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv51_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn51_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu51_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv52_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn52_{}'.format(i))
        conv_in = bn + conv_in
        conv_in = mx.sym.Activation(conv_in, act_type='relu', name='relu52_{}'.format(i))

    # 512x7x7
    # avg pooling, dense
    avg_pool = mx.sym.Pooling(conv_in, global_pool=True, kernel=(7,7), pool_type='avg')
    # 512x1x1
    weight = mx.sym.var('weight')
    flatten = mx.sym.flatten(avg_pool)
    scores = mx.sym.FullyConnected(flatten, weight=weight, num_hidden=cls_num, no_bias=True)

    # loss output for training
    softmax_output = mx.sym.SoftmaxOutput(scores, label=label)
    # loss
    loss, acc = SoftmaxLoss_with_Acc(scores, label)

    # return value
    weight_out = mx.sym.BlockGrad(weight)
    conv_in_out = mx.sym.BlockGrad(conv_in)
    loss_out = mx.sym.BlockGrad(loss)
    acc_out = mx.sym.BlockGrad(acc)
    img_out = mx.sym.BlockGrad(img)
    #  debug_out = mx.sym.BlockGrad(debug_out)
    out = mx.sym.Group([softmax_output, weight_out, conv_in_out, img_out, loss_out, acc_out])

    return out




if __name__ == '__main__':
    sym = resnet18(10)



