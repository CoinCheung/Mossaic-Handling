#!/usr/bin/python


import mxnet as mx
import core.symbol as symbol
import core.config as config
import core.DataIter as DI
import core.meric as meric


# get module
num_cls = config.cls_num
batch_size = config.batch_size
sym = symbol.resnet18(num_cls)

mod = mx.module.Module(
    sym,
    data_names=['img'],
    label_names=['label'],
    context=mx.gpu(0)
)
mod.bind(
    data_shapes=[('img', (batch_size, 3, 32, 32))],
    label_shapes=[('label', (batch_size, ))]
)
mod.init_params(mx.init.Xavier())
mod.init_optimizer(
    optimizer='adam',
    optimizer_params=(
        ('learning_rate', 1e-3),
        ('beta1', 0.9),
        ('wd', 5e-5),
    )
)



## get Data Iterator
train_iter, test_iter = DI.get_cifar10_iters()


## training process
loss = []
vacc = []
epoch = 10
i = 0
for e in range(epoch):
    for batch, label in train_iter:
        batch = mx.io.DataBatch([batch.as_in_context(mx.gpu())], label=[label.as_in_context(mx.gpu())])
        mod.forward(batch, is_train=True)
        mod.backward()
        mod.update()
        out = mod.get_outputs()
        scores = out[1].asnumpy()

        weight = out[2].asnumpy()



        if i % 50 == 0:
            acc = meric.acc(scores, label)
            vacc.append(acc)

        i += 1





