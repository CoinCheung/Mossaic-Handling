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



