#!/bin/bash

DIR=$HOME/.mxnet/datasets/

mkdir -p $DIR
wget -c -P $DIR http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf $DIR/VOCtrainval_11-May-2012.tar -C $DIR
rm $DIR/VOCtrainval_11-May-2012.tar
