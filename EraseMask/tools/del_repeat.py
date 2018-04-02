#!/usr/bin/python
# coding: utf-8

import os
import hashlib
import sys
from skimage import transform, io



def delete_repeat():
    path = '../dataset/Mossaic_JPEG/'
    images = os.listdir(path)
    org = [el for el in images if el.split('.')[-2].split('_')[-1] == 'org']
    org = [''.join([path, el]) for el in org]


    st = set([])
    def check_pair(fn):
        img = open(fn.encode('utf-8'), 'rb').read()
        m = hashlib.md5() # 这个不能多个共用
        m.update(img)
        md5_val = m.hexdigest()
        if not md5_val in st:
            st.add(md5_val)
        else:
            os.remove(fn)
            fn1 = fn.replace('org', 'mossaic')
            os.remove(fn1)
            print("remove: {}".format(fn))
            print("remove: {}".format(fn1))

    #  list(map(check_pair, org))


    images = [''.join([path, el]) for el in images]
    st = set([])
    def check_concat(fn):
        img = open(fn.encode('utf-8'), 'rb').read()
        m = hashlib.md5() # 这个不能多个共用
        m.update(img)
        md5_val = m.hexdigest()
        if not md5_val in st:
            st.add(md5_val)
        else:
            os.remove(fn)
            print("remove: {}".format(fn))

    list(map(check_concat, images))


if __name__ == "__main__":
    delete_repeat()
