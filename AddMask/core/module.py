#!/usr/bin/python



import mxnet as mx
import core.config as config
import symbol.symbol as symbol


def get_train_module():
    # control hyper parameters
    num_cls = config.cls_num
    batch_size = config.batch_size
    optimizer = config.optimizer
    lr_factor = config.lr_factor
    lr_steps = config.lr_steps
    lr_stop_val = config.lr_stop_val
    lr = config.learning_rate
    wd = config.weight_decay

    # get symbol
    sym = symbol.resnet18(num_cls)
    # module
    mod = mx.module.Module(
        sym,
        data_names=['img'],
        label_names=['label'],
        context=mx.gpu(0)
    )

    mod.bind(
        data_shapes=[('img', (batch_size, 32, 32, 3))],
        label_shapes=[('label', (batch_size, ))]
    )

    #  mod.init_params(mx.init.Xavier())
    #  mod.init_params(mx.init.Orthogonal())
    #  mod.init_params(mx.init.MSRAPrelu(slope=0))
    mod.init_params(mx.init.MSRAPrelu())

    #  lr_scheduler = mx.lr_scheduler.FactorScheduler(step=lr_steps, factor=lr_factor, stop_factor_lr=lr_stop_val)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[24e3, 40e3], factor=0.1)
    #  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[0, 4000, 6000, 8000], factor=0.1)
    if optimizer == "adam":
        mod.init_optimizer(
            optimizer=optimizer,
            optimizer_params=(
                ('learning_rate', lr),
                ('beta1', 0.9),
                ('wd', wd),
                ('lr_scheduler', lr_scheduler)
            )
        )
    elif optimizer == 'sgd':
        mod.init_optimizer(
            optimizer=optimizer,
            optimizer_params=(
                ('learning_rate', lr),
                ('momentum', 0.9),
                ('wd', wd),
                ('lr_scheduler', lr_scheduler)
            )
        )

    return mod


def get_test_module():
    batch_size = config.batch_size

    mod = mx.mod.Module.load(
        "./model_export/resnet18",
        0,
        True,
        context=mx.gpu(),
        data_names=['img'],
        label_names=['label'],
    )

    mod.bind(
        data_shapes=[('img',(batch_size,32,32,3))],
        label_shapes=[('label',(batch_size,))],
    )

    return mod




