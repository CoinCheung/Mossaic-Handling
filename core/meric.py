#!/usr/bin/python


import numpy as np



def accuracy(label_true, label_pred):
    label_true = label_true.reshape((-1,))
    label_pred = label_pred.reshape((-1,))
    acc = np.mean(label_true == label_pred)

    return acc



