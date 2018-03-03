#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config
import core.DataIter as DI
import core.meric as meric
import core.module as module
import core.visualize as visualize


## function that predict with given module and an iterator
def predict(mod, iterator, num_iters):
    pred_label = []
    true_label = []
    for (i, (batch,label)) in enumerate(iterator):
        if i < num_iters:
            databatch = mx.io.DataBatch([batch.as_in_context(mx.gpu())], label=[label.as_in_context(mx.gpu())])
            mod.forward(databatch,is_train=False)
            pred = mod.get_outputs()
            pred_label.append(pred[3].asnumpy().reshape((-1,)))
            true_label.append(pred[4].asnumpy().reshape((-1,)))
    pred_label = np.concatenate(pred_label, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    return pred_label, true_label



def train():
    ## get module used for training
    mod = module.get_train_module()

    ## get Data Iterator
    train_iter, test_iter = DI.get_cifar10_iters()

    ## training process
    training_loss = []
    valid_acc = []
    epoch = config.epoches
    for e in range(epoch):
        i = 0
        for batch, label in train_iter:
            # make data batch and train
            databatch = mx.io.DataBatch([batch.as_in_context(mx.gpu())], label=[label.as_in_context(mx.gpu())])
            mod.forward(databatch, is_train=True)
            mod.backward()
            mod.update()

            # get outputs
            out = mod.get_outputs()
            loss = out[2].asnumpy()
            training_loss.append(loss)

            #  weight = out[1].asnumpy()

            # valid the accuracy each 50 iterations and print training states
            if i % 50 == 0:
                pred_label, true_label = predict(mod, test_iter, 16)
                acc = meric.accuracy(true_label, pred_label)
                valid_acc.append(acc)
                print('epoch: {}, iters: {}, training loss: {}, valid accuracy: {}'
                    .format(e,i,loss,acc)
                    )

            i += 1

    # draw the training process
    valid_acc = np.array(valid_acc)
    training_loss = np.array(training_loss)
    visualize.draw_curve([valid_acc, training_loss], ['validation acc', 'training loss'])


if __name__ == "__main__":
    train()


