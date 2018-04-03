#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import mxnet.gluon as gluon



class UnetSkip(gluon.HybridBlock):
    def __init__(self, channels, in_channels=0, innermost=False, outermost=False, unit):
        super(UnetSkip, self).__init__()
        conv = gluon.nn.Conv2D(channels=channels, in_channels = in_channels,
                kernel_size=4, strids=(2,2), padding=(1,1))
        leaky = gluon.nn.LeakyReLU(alpha=0.2)
        down_BN = gluon.nn.BatchNorm(momentum=0.1, in_channels=channels)
        relu = gluon.nn.Activation(activation='relu')
        up_BN = gluon.nn.BatchNorm(momentum=0.1, in_channels=in_channels)
        deconv = gluon.nn.Conv2DTranspose(channels=in_channels, kernel_size=4,
                strides=(2,2), padding=(1,1))
        if innermost:
            encode =
            pass
        elif outermost:
            pass
        else:
            pass

    def forward(self, F, x):
        pass
