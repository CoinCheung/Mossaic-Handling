#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config
import symbol.meric as meric


def pre_processing(data, label):
    batch_size = config.batch_size

    # transpose and minus average
    data = mx.sym.transpose(data, axes=(0,3,1,2))
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




def resnet18(cls_num):
    img = mx.sym.var("img")
    label = mx.sym.var("label")

    # implement pre-processing
    img, label = pre_processing(img, label)

    # 3x32x32
    conv32 = mx.sym.Convolution(img, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False, name='conv1')
    # 16x32x32
    for i in range(3):
        conv = mx.sym.Convolution(conv32, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn2_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu2_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=16, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv3_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn3_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu3_{}'.format(i))
        conv32 = relu + conv32

    # 16x32x32
    conv = mx.sym.Convolution(conv32, num_filter=32, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv4_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn4_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu4_0')
    conv = mx.sym.Convolution(relu, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn5_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu5_0')
    shortcut = mx.sym.Convolution(conv32, num_filter=32, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut0')
    conv16 = relu + shortcut
    # 32x16x16
    for i in range(1,3):
        conv = mx.sym.Convolution(conv16, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv4_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn4_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu4_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn5_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu5_{}'.format(i))
        conv16 = relu + conv16

    # 32x16x16
    conv = mx.sym.Convolution(conv16, num_filter=64, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv6_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn6_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu6_0')
    conv = mx.sym.Convolution(relu, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv7_0')
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn7_0')
    relu = mx.sym.Activation(bn, act_type='relu', name='relu7_0')
    shortcut = mx.sym.Convolution(conv16, num_filter=64, kernel=(1,1), stride=(2,2), pad=(0,0), name='shortcut1')
    conv8 = relu + shortcut
    # 64x8x8
    for i in range(1,3):
        conv = mx.sym.Convolution(conv8, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv6_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn6_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu6_{}'.format(i))
        conv = mx.sym.Convolution(relu, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv7_{}'.format(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='bn7_{}'.format(i))
        relu = mx.sym.Activation(bn, act_type='relu', name='relu7_{}'.format(i))
        conv8 = relu + conv8

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





if __name__ == '__main__':
    sym = resnet18



