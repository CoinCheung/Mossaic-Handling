#!/usr/bin/python
# -*- encoding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import numpy as np




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
        #  sqrtn = int(np.ceil(np.sqrt(2*batch_size)))

        plt.figure(self.fig_num)
        #  gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs = gridspec.GridSpec(3, 2)
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

