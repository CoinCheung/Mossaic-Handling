#!/usr/bin/python


import mxnet as mx


def accuracy(label_true, label_pred):
    label_true = label_true.reshape((-1,))
    label_pred = label_pred.reshape((-1,))
    acc = mx.sym.mean(label_true == label_pred)

    return acc



