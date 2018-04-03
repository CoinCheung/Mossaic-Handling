1. The GAN is designed mainly following the principles in this paper


pix2pix feature:
1. conditional gan structure. use original image concated images to feed the discriminator
2. unet, input is a masked image with no condition
3. patchnet is a conditional gan, with masked (or input of gen) to be condition
4. noise is introduced by dropout 

5. use only patched logistic loss to train discriminator, while use a combination of L1 loss and logistic loss to train generator
