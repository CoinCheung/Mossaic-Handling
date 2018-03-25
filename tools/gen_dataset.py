#!/usr/bin/python


import random
import os


root_dir = os.getcwd() + '/../'
dataset_dir = root_dir + 'dataset/'
img_dir = dataset_dir + 'JPEG_IMG/'


def get_random_split():
    pass

def gen_lst():
    img_fns = os.listdir(img_dir)
    img_paths = [''.join([img_dir, fn]) for fn in img_fns]
    print(img_paths[:10])


if __name__ == "__main__":
    gen_lst()
