#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import mxnet.gluon as gluon
import os
import numpy as np

import core.config as config
from core.DataIter import get_dataiter, img_norm
from core.Visualize import ImgGrids
from core.Network import get_network


epoch = config.epoch
batch_size = config.batch_size
save_path = config.save_path


def train():
    gen, dis, TrainerG, TrainerD = get_network()
    lossGAN = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    lossL1 = gluon.loss.L1Loss()

    train_it, val_it = get_dataiter()
    ImgHandler = ImgGrids(1)

    count = 0
    lossG_arr = []
    lossD_arr = []
    for e in range(epoch):
        train_it.reset()
        for batch in train_it:
            data_org, data_masked = mx.nd.split(batch.data[0], axis=3, num_outputs=2)

            data_org = img_norm(data_org).as_in_context(mx.gpu())
            data_masked = img_norm(data_masked).as_in_context(mx.gpu())
            data_gen = gen(data_masked)
            data_gen = data_gen.as_in_context(mx.gpu())

            ## train dis
            with mx.autograd.record():
                # real side
                dis_batch = mx.nd.concat(data_org, data_masked, dim=1)
                dis_out = dis(dis_batch)
                label = mx.nd.ones_like(dis_out)
                lossD_real = lossGAN(dis_out, label)
                # fake side
                dis_batch = mx.nd.concat(data_gen, data_masked, dim=1)
                dis_out = dis(dis_batch)
                label = mx.nd.zeros_like(dis_out)
                lossD_fake = lossGAN(dis_out, label)
                # loss in total
                lossD = (lossD_real + lossD_fake) * 0.5
                # compute gradient
                lossD.backward()
                lossD_arr.append(np.mean(lossD.asnumpy()))
            TrainerD.step(batch_size)

            ## train gen
            with mx.autograd.record():
                # get gen batch data
                data_gen = gen(data_masked)
                data_gen = data_gen.as_in_context(mx.gpu())
                gen_batch = mx.nd.concat(data_gen, data_masked, dim=1)
                # compute loss
                dis_out = dis(gen_batch)
                label = mx.nd.ones_like(dis_out)
                lossG_GAN = lossGAN(dis_out, label)
                lossG_L1 = lossL1(data_gen, data_org)
                # loss in total
                lossG = lossG_GAN + 100 * lossG_L1
                lossG.backward()
                lossG_arr.append(np.mean(lossG.asnumpy()))
                # compute
            TrainerG.step(batch_size)

            count += 1
            if count % 20 == 0:
                print("epoch: {}, iter: {}, lossG: {}, lossD: {}".format(e, count,
                    lossG_arr[-1], lossD_arr[-1]))
                img_show = [data_masked[:2].asnumpy(), data_gen[:2].asnumpy(),
                        data_org[:2].asnumpy()]
                ImgHandler.draw(img_show)

    gen.export(save_path + "/generator")
    dis.export(save_path + "/discriminator")


if __name__ == "__main__":
    train()
