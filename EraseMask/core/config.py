#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os


optimizer = 'adam'
lr = 2e-4
beta1 = 0.5
epoch = 10
batch_size = 9
save_path = os.getcwd() + "/model_export/"

# test parameters
test_batch_size = 16
generate_batch_num = 5
