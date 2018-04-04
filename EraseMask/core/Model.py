#!/usr/bin/python
# -*- encoding: utf-8 -*-


import mxnet as mx
import mxnet.gluon as gluon



### models
## generator unet block
class UnetSkipBlock(gluon.nn.HybridBlock):
    def __init__(self, inner_channels, outer_channels, innermost=False,
            outermost=False, use_dropout=False, unit=None):
        super(UnetSkipBlock, self).__init__()

        self.outermost = outermost
        with self.name_scope():
            conv = gluon.nn.Conv2D(channels=inner_channels, in_channels=outer_channels,
                    kernel_size=4, strides=(2,2), padding=(1,1))
            leaky = gluon.nn.LeakyReLU(alpha=0.2)
            down_BN = gluon.nn.BatchNorm(momentum=0.1, in_channels=inner_channels)
            relu = gluon.nn.Activation(activation='relu')
            up_BN = gluon.nn.BatchNorm(momentum=0.1, in_channels=outer_channels)
            tanh = gluon.nn.Activation(activation='tanh')
            dropout = gluon.nn.Dropout(0.5)

            if innermost:
                deconv = gluon.nn.Conv2DTranspose(channels=outer_channels,
                        in_channels=inner_channels, kernel_size=4, strides=(2,2),
                        padding=(1,1))
                down = [leaky, conv]
                up = [relu, deconv, up_BN]
                model = down + up
            elif outermost:
                deconv = gluon.nn.Conv2DTranspose(channels=outer_channels,
                        in_channels=2*inner_channels, kernel_size=4, strides=(2,2),
                        padding=(1,1))
                down = [conv]
                up = [relu, deconv, tanh]
                model = down + [unit] + up
            else:
                deconv = gluon.nn.Conv2DTranspose(channels=outer_channels,
                        in_channels=2*inner_channels, kernel_size=4, strides=(2,2),
                        padding=(1,1))
                down = [leaky, conv, down_BN]
                up = [relu, deconv, up_BN]
                model = down + [unit] + up

            if use_dropout:
                model += [dropout]

            self.mod = gluon.nn.HybridSequential()
            with self.mod.name_scope():
                for layer in model:
                    self.mod.add(layer)
            #  self.mod.hybridize()

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.mod(x)
        else:
            return F.concat(self.mod(x), x, dim=1)



## Unet generator HybridBlock implementation
class gen_unet(gluon.nn.HybridBlock):
    def __init__(self, in_channels, filter_base=64, use_dropout=True, num_down=5):
        super(gen_unet, self).__init__()

        # inner most layer
        unet = UnetSkipBlock(filter_base*8, filter_base*8, innermost=True)
        # middle layers with the most layers to be 8xfilter_base
        for i in range(num_down - 5):
            unet = UnetSkipBlock(filter_base*8, filter_base*8, unit=unet,
                    use_dropout=use_dropout)
        unet = UnetSkipBlock(filter_base*8, filter_base*4, unit=unet)
        unet = UnetSkipBlock(filter_base*4, filter_base*2, unit=unet)
        unet = UnetSkipBlock(filter_base*2, filter_base, unit=unet)
        # outer most layer
        unet = UnetSkipBlock(filter_base, in_channels, unit=unet, outermost=True)

        with self.name_scope():
            self.gen_model = unet

    def hybrid_forward(self, F, x):
        return self.gen_model(x)



def generator_unet(in_channels, filter_base=64, use_dropout=True, num_down=5):
    # inner most layer
    unet = UnetSkipBlock(filter_base*8, filter_base*8, innermost=True)
    # middle layers with the most layers to be 8xfilter_base
    for i in range(num_down - 5):
        unet = UnetSkipBlock(filter_base*8, filter_base*8, unit=unet,
                use_dropout=use_dropout)
    unet = UnetSkipBlock(filter_base*8, filter_base*4, unit=unet)
    unet = UnetSkipBlock(filter_base*4, filter_base*2, unit=unet)
    unet = UnetSkipBlock(filter_base*2, filter_base, unit=unet)
    # outer most layer
    unet = UnetSkipBlock(filter_base, in_channels, unit=unet, outermost=True)

    gen_model = gluon.nn.HybridSequential()
    with gen_model.name_scope():
        gen_model.add(unet)

    return gen_model


def discriminator(in_channels, filter_base=64, nlayer=3):
    model = gluon.nn.HybridSequential()
    with model.name_scope():
        conv = gluon.nn.Conv2D(channels=filter_base, in_channels=in_channels,
            kernel_size=(4,4), strides=(2,2), padding=(1,1))
        leaky = gluon.nn.LeakyReLU(0.2)
        model.add(conv)
        model.add(leaky)

        filter_num = filter_base
        for i in range(1, nlayer):
            filter_prev = filter_num
            filter_num = min(filter_base*8, filter_prev*2)
            conv = gluon.nn.Conv2D(channels=filter_num, in_channels=filter_prev,
                kernel_size=(4,4), strides=(2,2), padding=(1,1))
            bn = gluon.nn.BatchNorm(momentum=0.1, in_channels=filter_num)
            leaky = gluon.nn.LeakyReLU(0.2)
            model.add(conv)
            model.add(bn)
            model.add(leaky)

        filter_prev = filter_num
        filter_num = min(filter_base*8, filter_prev*2)
        conv = gluon.nn.Conv2D(channels=filter_num, in_channels=filter_prev,
            kernel_size=(4,4), strides=(1,1), padding=(1,1))
        bn = gluon.nn.BatchNorm(momentum=0.1, in_channels=filter_num)
        leaky = gluon.nn.LeakyReLU(0.2)
        out = gluon.nn.Conv2D(channels=1, in_channels=filter_num,
            kernel_size=(4,4), strides=(1,1), padding=(1,1))
        model.add(conv)
        model.add(bn)
        model.add(leaky)
        model.add(out)

    return model
