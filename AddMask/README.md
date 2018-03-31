# Add Mossaic

This program is designed to add mossaic to images. The mossaic is added to the class activation maps (CAM) of the images which is determined by a resnet trained for classification. More details of CAM is available at: **https://arxiv.org/abs/1512.04150**

The platform for this program to run properly is linux. It also requires opencv and mxnet python api installed.

## Create dataset
Make sure the dataset is created before all other operations:
```sh
    $ cd Mossaic-Handling/tools
    $ sh gen_dataset.sh
```
The dataset is actually a portion of Imagenet-1000. Just enough images are selected to train the resetnet.


## Training
The resnet, or resnet-18 exactly, should be trained first before it is used to find CAMs of the images. Run the script to train the resnet-18:
```sh
    $ cd Mossaic-Handling/AddMask
    $ python train.py
```
Since the dataset is a small one. One will see the cross validation accuracy roars up above 90% within a few epoches of training.


## Generating Masked Images
The part of adding mossaic to a given image according to its CAM is written with C++. So one should build the C++ program first:
```
    $ cd Mossaic-Handling/AddMask/Mask
    $ make
```
Since this program relies on opencv, one should install opencv libs before build the program.   
After that one could use the trained model to generate the images with mossaic:
```sh
    $ cd Mossaic-Handling/AddMask
    $ python gen_image.py
```
After the program is done, one might see the images and its masked counterpart in the directory of ```Mossaic-Handling/AddMask/pictures_export```.


## Tips

1. The model is trained on 224x224 images, so it should work well on 224x224 test images. If someone would like to use this model on images of other resolutions, he or she could resize the images before feeding them to the model.

2. to adjust the ratio of the mossaic added to the orginal picture, modify the value of the variable ```mossaic_ratio``` in the file ```AddMask/core/config.py```.

3. one may choose whether heat map images should be generated along with the masked images through modifying the boolean value of the variable ```gen_heatmap``` in the file ```AddMask/core/config.py```.
