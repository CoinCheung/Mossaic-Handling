#!/bin/bash



# paths
root_dir=../
img_dir=$root_dir/dataset/Mossaic_JPEG
lst_dir=$root_dir/dataset
dataset_pth=$HOME/.mxnet/datasets/MaskDataSet/Erase/


# generate lst files
python3 gen_lst.py

# generate rec file
python3 ./im2rec.py --train-ratio=1 --test-ratio=0 --pack-label $lst_dir $img_dir

# collect the dataset files
mkdir -p $dataset_pth
mv -ivu $lst_dir/*.lst $dataset_pth/
mv -ivu $lst_dir/*.idx $dataset_pth/
mv -ivu $lst_dir/*.rec $dataset_pth/
