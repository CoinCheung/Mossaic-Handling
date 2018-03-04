#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab



def draw_curve(data_list, titles, fig_num=1):
    subnum = len(data_list)

    _, ax = plt.subplots(subnum,1, num=fig_num, sharex=False,sharey=False)
    plt.ioff()
    for sn in range(subnum):
        x = np.arange(data_list[sn].shape[0])
        y = data_list[sn]
        ax[sn].plot(x,y)
        ax[sn].set_title(titles[sn])
    plt.tight_layout()
    plt.show()



class ImgGrids(object):
    '''
    TODO: add annotations
    '''
    def __init__(self, fig_num):
        self.fig_num = fig_num
        self.fig = plt.figure(fig_num)
        plt.ion()

    def draw(self, img):
        if type(img) == np.ndarray:
            batch_size, c, w, h = img.shape
        elif type(img) == list:
            batch_size = len(img)
            c = 1
            w, h = img[0].shape

        sqrtn = int(np.ceil(np.sqrt(batch_size)))

        plt.figure(self.fig_num)
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)

        for i,image in enumerate(img):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            if c == 3 or c == 4:
                pylab.imshow(image.transpose(1,2,0))
            else:
                plt.imshow(image.reshape(w,h))
        plt.show()
        plt.pause(0.001)

