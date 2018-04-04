#!/usr/bin/python


import mxnet as mx
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import numpy as np


batch_size = 15
epoch = 10
image_shape = (batch_size, 3, 224, 224)
lr = 2e-4
beta1 = 0.5
wd = 0
save_path = os.getcwd() + "/model_export/"


## symbols
def generator():
    slope = 0.2
    filter_base = 64
    filter_max = filter_base * 8

    img_masked = mx.sym.var('img_masked')
    img_org = mx.sym.var('img_org')
    ### unet
    filter_nums = [filter_base]
    downs = []
    ## encoding
    # 3x224x224
    conv = mx.sym.Convolution(img_masked, num_filter=filter_base, kernel=(4,4), stride=(2,2), pad=(1,1), name='gen_conv0')
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
    img_gen = mx.sym.Activation(dconv, act_type='tanh', name='img_gen')
    # 3x224x224

    L1 = mx.sym.mean(img_gen - img_org, axis=(1,2,3))
    L1 = mx.sym.MakeLoss(L1)

    out = mx.sym.Group([img_gen, L1])

    return out


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
    mod = mx.mod.Module(sym, context=mx.gpu(),
            data_names=['img_masked', 'img_org'], label_names=None)
    mod.bind(data_shapes=[('img_masked', image_shape), ('img_org', image_shape)])
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
            label_shapes=[('label', (batch_size, 1)),], inputs_need_grad=True)
    mod.init_params(mx.init.Normal(0.02))
    mod.init_optimizer(
            optimizer='adam', optimizer_params=(('learning_rate', lr),
                ('beta1', 0.5), ('wd', wd), ))

    return mod


### data iterators
def get_dataiter():
    home_dir = os.path.expandvars('$HOME')
    train_path = home_dir + '/.mxnet/datasets/MaskDataSet/Erase/train.rec'
    val_path = home_dir + '/.mxnet/datasets/MaskDataSet/Erase/val.rec'
    seed = random.randint(0, 5000)

    img_shape = (3, 224, 448)
    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_path,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
    )
    val_iter = mx.io.ImageRecordIter(
        path_imgrec=val_path,
        data_shape=img_shape,
        label_width=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,
    )

    return train_iter, val_iter

def img_norm(img):
    max_ = mx.nd.max(img)
    min_ = mx.nd.min(img)
    img = (2 * img - max_ - min_) / (max_ - min_)
    return img


### visualization
class ImgGrids(object):
    def __init__(self, fig_num):
        self.fig_num = fig_num
        self.fig = plt.figure(fig_num)
        plt.ion()

    def process(self, im):
        im = np.transpose(im, (1,2,0))
        max_ = np.max(im)
        min_ = np.min(im)
        im = (im - min_) / (max_ - min_)
        #  print(max_)
        #  print(min_)
        return im

    def draw(self, img):
        batch_size, c, w, h = img[0].shape
        sqrtn = int(np.ceil(np.sqrt(2*batch_size)))

        plt.figure(self.fig_num)
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)

        i = 0
        for im_pair in img:
            for image in im_pair:
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                image = self.process(image)
                plt.imshow(image)
                i += 1
        plt.show()
        plt.pause(0.001)


### training precedure
def train():

    gen = get_module_gen_train()
    dis = get_module_dis()

    it_train, it_val = get_dataiter()

    ImgHd = ImgGrids(1)

    # train discriminator
    genloss = []
    disloss = []
    count = 0
    for e in range(epoch):
        it_train.reset()
        for batch in it_train:
            data = batch.data[0]
            data_org, data_masked = mx.nd.split(data, axis=3, num_outputs=2)
            data_org = img_norm(data_org)
            data_masked = img_norm(data_masked)

            gen_batch = mx.io.DataBatch(data=[data_masked, data_org], label=None)
            gen.forward(gen_batch)
            gen_loss_L1 = gen.get_outputs()[1].asnumpy()
            data_fake = gen.get_outputs()[0]

            ## train discriminator
            ### train on generated images
            label_fake = mx.nd.zeros((batch_size, 1), ctx=mx.gpu())
            dis_batch = mx.io.DataBatch(data=[data_fake, data_masked], label=[label_fake])
            dis.forward(dis_batch)
            dis_loss1 = dis.get_outputs()[1].asnumpy()
            dis.backward()
            grad_real = [[grad.copyto(grad.context) for grad in grads]
                         for grads in dis._exec_group.grad_arrays]

            ### train on original images
            label_real = mx.nd.ones((batch_size, 1), ctx=mx.gpu())
            dis_batch = mx.io.DataBatch(data=[data_org, data_masked], label=[label_real])
            dis.forward(dis_batch)
            dis_loss2 = dis.get_outputs()[1].asnumpy()
            dis.backward()
            def grad_add(g1, g2):
                g1 += g2
                g1 /= 2
            def grad_list_add(gl1, gl2):
                list(map(grad_add, gl1, gl2))
            list(map(grad_list_add, dis._exec_group.grad_arrays, grad_real))
            dis.update()
            dis_loss = (dis_loss1 + dis_loss2)/2
            disloss.append(dis_loss)

            ## train generator
            label_real = mx.nd.ones((batch_size, 1), ctx=mx.gpu())
            dis_batch = mx.io.DataBatch(data=[data_fake, data_masked], label=[label_real])
            dis.forward(dis_batch)
            gen_loss_GAN = dis.get_outputs()[1].asnumpy()
            dis.backward()
            L1_grad = 100 * mx.nd.ones((batch_size, ), ctx=mx.gpu())
            dis_grad = dis.get_input_grads()
            dis_grad.append(L1_grad)
            gen.backward(dis_grad)
            gen.update()
            gen_loss = np.mean(gen_loss_L1) + gen_loss_GAN
            genloss.append(gen_loss)

            count += 1
            if count % 10 == 0:
                print("epoch: {}, iter: {}, gen_loss: {}, dis_loss: {}".format(e,
                    count, genloss[-1], disloss[-1]))
                img_show = [data_fake[:2].asnumpy(), data_org[:2].asnumpy()]
                ImgHd.draw(img_show)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dis.save_checkpoint(save_path + "discriminator", 0, True)
    gen.save_checkpoint(save_path + "generator", 0, True)




if __name__ == '__main__':
    train()
