#!/usr/bin/python


import mxnet as mx


batch_size = 64
image_shape = (batch_size, 3, 224, 224)
lr = 2e-4
beta1 = 0.5
wd = 0


## symbols


def generator():
    slope = 0.2
    filter_base = 64
    filter_max = filter_base * 8

    img = mx.sym.var('img')
    ### unet
    filter_nums = [filter_base]
    downs = []
    ## encoding
    # 3x224x224
    conv = mx.sym.Convolution(img, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv0')
    downs.append(conv)
    relu = mx.sym.LeakyReLU(conv, act_type='leaky', slope=slope, name='gen_leaky0')
    # 64x112x112
    for i in range(1, 4):
        filter_base *= 2
        filter_base = min(filter_base, filter_max)
        filter_nums.append(filter_base)
        conv = mx.sym.Convolution(relu, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv'+str(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='gen_down_bn'+str(i))
        relu = mx.sym.LeakyReLU(bn, act_type='leaky', slope=slope, name='gen_leaky'+str(i))
        downs.append(bn)
    # 512x14x14
    filter_base *= 2
    filter_base = min(filter_base, filter_max)
    conv = mx.sym.Convolution(relu, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv'+str(4))
    # 1024x7x7

    # decoding
    join = conv
    for i in range(1, 5):
        filter_base = filter_nums[0-i]
        relu = mx.sym.Activation(join, act_type='relu', name='gen_relu'+str(i))
        dconv = mx.sym.Deconvolution(relu, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=filter_base, name='gen_dconv'+str(i))
        bn = mx.sym.BatchNorm(dconv, fix_gamma=False, name='gen_up_bn'+str(i))
        dp = mx.sym.Dropout(bn, p=0.5, mode='always', name='gen_up_dp'+str(i))
        join = mx.sym.concat(dp, downs[0-i], dim=1)
    # 64x112x112
    relu = mx.sym.Activation(join, act_type='relu', name='gen_relu'+str(5))
    dconv = mx.sym.Deconvolution(relu, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, name='dconv_last')
    img_gen = mx.sym.Activation(join, act_type='tanh', name='img_gen')
    # 3x224x224

    return img_gen


def discriminator():
    imgA = mx.sym.var('out_img')
    imgB = mx.sym.var('original_img')
    label = mx.sym.var('label')
    img = mx.sym.concat(imgA, imgB, dim=1)

    nlayers = 4
    filter_base = 64
    filter_mul = 1
    slope = 0.2

    # 6x224x224
    conv = mx.sym.Convolution(img, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='dis_conv0')
    relu = mx.sym.LeakyReLU(conv, act_type='leaky', slope=slope, name='dis_leaky0')
    # 64x112x112
    for i in range(1, nlayers):
        filter_mul *= 2
        filter_mul = min(filter_mul, 8)
        conv = mx.sym.Convolution(relu, num_filter=filter_base*filter_mul, kernel=(4,4), stride=(2,2), pad=(1,1), name='dis_conv'+str(i))
        bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='dis_bn'+str(i))
        relu = mx.sym.LeakyReLU(bn, act_type='leaky', slope=slope, name='dis_leaky'+str(i))
    # 512x14x14
    filter_mul = 2**nlayers
    conv = mx.sym.Convolution(relu, num_filter=filter_base*filter_mul, kernel=(4,4), stride=(2,2), pad=(1,1), name='dis_conv'+str(nlayers))
    bn = mx.sym.BatchNorm(conv, fix_gamma=False, name='dis_bn'+str(nlayers))
    relu = mx.sym.LeakyReLU(bn, act_type='leaky', slope=slope, name='dis_leaky'+str(nlayers))
    # (2**nlayers)x 7x7
    conv = mx.sym.Convolution(relu, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), name='dis_conv_out')
    # batch_size x 1 x 7 x 7

    flatten = mx.sym.Flatten(conv)
    label = mx.sym.broadcast_to(label, shape=(batch_size, 49))
    sigmoid = mx.sym.sigmoid(flatten)
    loss_out = mx.sym.BlockGrad(mx.sym.mean(mx.sym.log(sigmoid+1e-12)*label))
    loss = mx.sym.LogisticRegressionOutput(data = flatten, label = label)

    out = mx.sym.Group([loss, loss_out])

    return out


### modules
def get_module_gen_train():
    sym = generator()
    mod = mx.mod.Module(sym, context=mx.gpu(), data_names=['img'], label_names=None)
    mod.bind(data_shapes=[('img', image_shape)])
    mod.init_params(mx.init.Normal(0.02))
    mod.init_optimizer(
            optimizer='adam', optimizer_params=(('learning_rate', lr),
                ('beta1', 0.5), ('wd', wd), ))

    return mod


def get_module_dis():
    sym = discriminator()
    mod = mx.mod.Module(sym, context=mx.gpu(),
            data_names=['out_img', 'original_img'], label_names=['label'])
    mod.bind(data_shapes=[('out_img', image_shape), ('original_img', image_shape)],
            label_shapes=[('label', (batch_size, 1)),])
    mod.init_params(mx.init.Normal(0.02))
    mod.init_optimizer(
            optimizer='adam', optimizer_params=(('learning_rate', lr),
                ('beta1', 0.5), ('wd', wd), ))

    return mod


### data iterators
def get_dataiter():
    pass


def train():
    pass

    gen = get_module_gen_train()
    dis = get_module_dis()

    it = get_DataIter()


if __name__ == '__main__':
    train()
