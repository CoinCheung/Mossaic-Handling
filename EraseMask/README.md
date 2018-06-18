# Erase Mossaic
This program is designed to remove mossaic from images. The algorithm used to do this task is pix2pix GAN, the principles can be found in this paper: https://arxiv.org/abs/1611.07004.

The main idea of this program is as typical GAN implementation: first we train the pix2pix GAN with pairs of masked images and their associated original images, then we use only generator of the GAN to "translate" the masked images to their unmasked counterparts. What is worth mentioning is that since adding mossaic to images is a process of losing information. Namely, some high-frequency components are removed from the images when mossaic is introduced. Obviously, it is not possible to remove the mossaic such that the derived images look exactly like their unmasked counterparts since we have no way to supplement the missing information of the images without knowing what information is missing given merely masked images. The generator of a GAN does introduce some information to the images so that the derived images look unmasked, but the introduced information is not the same as what is missing from the original images. A GAN provides these informations according to its experience of seeing so many training images, that is to say, a GAN "imagines" what the unmasked images should be like and add informations to the mossaic areas to make the derived images look like those in its imagination. Therefore, one should be careful to the senarios where a pix2pix GAN should be used.


## Platform and Environments
This program is developed and should be run on the platform of Linux with python3.6 installed.


## Creating datasets
I have prepared the training and testing datasets with the method introduced in the ```AddMask``` directory. The raw images could be found in the directory of ```EraseMask/dataset/Mossaic_JPEG/```. To generate the record files for mxnet/gluon, one could run the script:
```shell
    $ cd tools
    $ sh gen_dataset.sh
```
After a few minutes, the record will be generated.


## Training
Run the ```train.py``` program to start training the network.
```shell
    $ python train.py
```
One can change the batch size and optimizer parameters to control the training process accroding his or her environments. These changes could be done in the file of ```core/config.py```.  


## Generate unmasked images
After training, one could test the network to see what a work it has done. Run the program:
```shell
    $ python gen_images.py
```
The one could see the orignal images, masked images and the mossaic-removed images in the directory of ```pics_export/```.



## Tips in implementation
Here are some details of how to implement this pix2pix GAN, which I found worth noticing:  

1. **Conditional GAN discriminator:** When training the network, masked images are concated to their "real" or "fake" images counterparts as a condition to feed into the discriminator. Only discriminator needs this condition, generator receives normal image input.

2. **Unet generator:** The generator is an encoder-decoder structure with symmetric layers concated together apart from the outer-most and inner-most layers. More details could be found in the paper.

3. **Patchnet discriminator:** Rather than a sigmoid single value, the discriminator outputs a sigmoid feature map without average pooling operations. Each point on the last feature map is associated with an area of patch on the input image. Pass the point through a sigmoid function to deduce the probablity whether this patch on the input is a true patch or a generated patch. The label (true or generated) should be broadcasted to the shape same as the last feature map so that a cross-entropy loss is computed for backward propagation.

4. **Dropout noise:** In the classical GAN, random noises are added to the generator as input. For pix2pix GAN, noise is still needed, but they are introduced by adding dropout layer to the innner layers whose channel numbers are 512.

5. **Loss:** We use only patched logistic loss to train discriminator, while we use a combination of L1 loss (between the generated images and their original counterparts) and patched logistic loss to train generator
