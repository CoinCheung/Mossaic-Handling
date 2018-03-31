# Mossaic-Handling
A program that can find the sensitive areas in a picture and add Mossaic to these areas automatically

<!-- The platform is linux, since multi-processing is used to load data and mxnet.gluon does not support multi-processing on windows platform -->


## get the dataset
Run the script in the project root directory:
```sh
    $ sh scripts/get_dataset.sh
```
This will download the VOC2012 dataset to directory ~/.mxnet/datasets/VOC2012



# Training Resnet-18 on Cifar-10 datasets
Just run the script of train.py
```
    $ python train.py
```
Then the resnet-18 network will be trained on Cifar-10 dataset.


# Test and generate the heat maps
Run the test.py script, and images together with their associated heat map images will be saved to the directory of "pictures_export"
```sh
    $ python test.py
```

# build the add mossaic C++ program
```sh
    $ cd AddMask/Mask
    $ make clean && make
```
This requres opencv installed ahead of time for link.


1. The model is trained on 224x224 images, so it should work well on 224x224 test images. If someone would like to use this model on images of other resolutions, he or she could resize the images before feeding them to the model.

2. to adjust the ratio of the mossaic added to the orginal picture, modify the value of the variable ```mossaic_ratio``` in the file ```AddMask/core/config.py```.
