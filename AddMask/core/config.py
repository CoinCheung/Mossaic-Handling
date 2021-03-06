

## Work for cifar-10 dataset
#  batch_size = 128
#  cls_num = 10
#
#  image_size = 32
#  epoches = 1
#
#
#  # optimizer params
#  weight_decay = 1e-4
#  lr_factor = 0.1
#  lr_steps = 3000
#  lr_stop_val = 1e-7
#  learning_rate = 1e-3
#
#  optimizer = 'adam'
#


generate_batches = 5
mossaic_ratio = 0.3
gen_heatmap = False
save_path = "./pictures_export/"


####
## Work for imagenet dataset
cls_num = 3
layer_num = 18
batch_size = 48
image_size = 224

epoches = 70

# optimizer params
weight_decay = 1e-4
lr_factor = 0.1
lr_steps = 7000
lr_stop_val = 1e-7
learning_rate = 1e-1

#  optimizer = 'adam'
optimizer = 'sgd'

