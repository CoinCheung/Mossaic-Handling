#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import mxnet.gluon as gluon
from core.Model import *
import core.config as config



batch_size = config.batch_size
optimizer = config.optimizer
lr = config.lr
beta1 = config.beta1



### initializing the networks
def network_init(net):
    for param in net.collect_params().values():
        if param.name.find('conv') != -1:  # if it is conv layer
            if param.name.find('weight') != -1:
                param.initialize(init=mx.init.Normal(0.02), ctx=mx.gpu())
            else:
                param.initialize(init=mx.init.Zero(), ctx=mx.gpu())
        elif param.name.find('batchnorm') != -1:
            param.initialize(init=mx.init.Zero(), ctx=mx.gpu())
            if param.name.find('gamma') != -1:
                param.set_data(mx.nd.random_normal(1, 0.02, param.data().shape))




## get networks and their associated trainers
def get_train_network():
    gen = generator_unet(3, 64, True, 5)
    #  gen = gen_unet(3, 64, True, 5)
    gen.hybridize()
    dis = discriminator(6, 64, 3)
    dis.hybridize()

    network_init(gen)
    network_init(dis)

    TrainerG = gluon.Trainer(gen.collect_params(), optimizer,
            {'learning_rate': lr, 'beta1': beta1})
    TrainerD = gluon.Trainer(dis.collect_params(), optimizer,
            {'learning_rate': lr, 'beta1': beta1})

    return gen, dis, TrainerG, TrainerD



def get_test_network():
    sym = mx.sym.load(config.save_path + '/generator-symbol.json')
    net = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu())
    net.bind(data_shapes=[('data', (batch_size, 3, 224, 224))])
    net.load_params(config.save_path + '/generator-0000.params')

    return net

