#!/usr/bin/python
# -*- encoding = utf-8 -*-


import random
import os


root_dir = os.getcwd() + '/../'
dataset_dir = root_dir + 'dataset/'
img_dir = dataset_dir + 'Mossaic_JPEG/'



def gen_lst():
    img_fns = set(os.listdir(img_dir))
    val_fns = set(random.sample(img_fns, len(img_fns)//5))
    set(map(img_fns.remove, val_fns))
    train_fns = img_fns
    print(len(val_fns))
    print(len(train_fns))

    def write_one_item(fn, fhandle, ind):
        #  cls = fn.split('_')[0]
        #  if cls == 'fish':
        #      label = 0
        #  elif cls == 'bird':
        #      label = 1
        #  else:
        #      label = 2
        line = '{}\t{}\t{}\n'.format(ind, 0, fn)
        fhandle.write(line)

    with open(dataset_dir + 'train.lst', 'w') as tf:
        [write_one_item(fn, tf, i) for i, fn in enumerate(train_fns)]

    with open(dataset_dir + 'val.lst', 'w') as vf:
        [write_one_item(fn, vf, i) for i, fn in enumerate(val_fns)]


if __name__ == "__main__":
    gen_lst()
