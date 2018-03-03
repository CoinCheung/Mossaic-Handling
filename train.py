#!/usr/bin/python


import mxnet as mx
import numpy as np
import core.config as config
import core.DataIter as DI
import core.module as module
import core.visualize as visualize


## function that predict with given module and an iterator
def predict(mod, iterator, num_iters):
    feature_map = []
    acc_out = []
    for (i, (batch,label)) in enumerate(iterator):
        if i < num_iters:
            databatch = mx.io.DataBatch([batch.as_in_context(mx.gpu())], label=[label.as_in_context(mx.gpu())])
            mod.forward(databatch,is_train=False)
            pred = mod.get_outputs()
            feature_map.append(pred[2].asnumpy())
            acc_out.append(pred[-1].asnumpy().reshape((-1,)))
    acc_out = np.mean(np.concatenate(acc_out, axis=0))
    return feature_map, acc_out



def train():
    # see changes of lr
    count = 0
    lr = config.learning_rate
    lr_factor = config.lr_factor
    lr_steps = config.lr_steps

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
            loss = out[-2].asnumpy()
            training_loss.append(loss)

            # valid the accuracy each 50 iterations and print training states
            if i % 50 == 0:
                _, acc_out = predict(mod, test_iter, 16)
                valid_acc.append(acc_out)
                print('epoch: {}, iters: {}, training loss: {}, valid accuracy: {}'
                    .format(e,i,loss,acc_out)
                    )

            i += 1

            # follow lr changes
            count += 1
            if count % lr_steps == 0:
                lr *= lr_factor
                print("lr becomes: {}".format(lr))

    # export the trained model for future usage
    mod.save_checkpoint("./model_export/reset18", 0, True)

    # draw the training process
    valid_acc = np.array(valid_acc)
    training_loss = np.array(training_loss)
    visualize.draw_curve([valid_acc, training_loss], ['validation acc', 'training loss'])


if __name__ == "__main__":
    train()


