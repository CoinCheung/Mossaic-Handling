#!/usr/bin/python
# -*- encoding: utf8 -*-


import ctypes


lib = ctypes.cdll.LoadLibrary('./Mask/lib/libmossaic.so')

def add_mossaic_fun():
    add_mask = lib.add_mossaic
    add_mask.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_float]

    return add_mask


def gen_mossaic_dataset():
    gen_mossaic = lib.generate_AB_dataset
    gen_mossaic.argtypes = (ctypes.c_char_p, ctypes.c_float)

    return gen_mossaic
