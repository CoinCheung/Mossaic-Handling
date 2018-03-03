#!/usr/bin/python
# -*- encoding: utf8 -*-


import mxnet as mx
import core.module as module
import core.DataIter as DI
import core.config as config



def test():
    batch_num = config.generate_batches
    # get trained module
    mod = module.get_test_module()
    # get dataIter
    _, it = DI.get_cifar10_iters()

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
        weight = weight.append(out[1].asnumpy())
        feature_maps.append(out[2].asnumpy())

        hm = heat_maps(weight, feature_maps)
        img_save_to(hm, batch.asnumpy(), str(count))

        count += 1



if __name__ == "__main__":
    test()

