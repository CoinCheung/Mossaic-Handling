# Mossaic-Handling
A program that can find the sensitive areas in a picture and add Mossaic to these areas automatically

The platform is linux, since multi-processing is used to load data and mxnet.gluon does not support multi-processing on windows platform

# Training Resnet-18 on Cifar-10 datasets
Just run the script of train.py
```
    $ python train.py
```
Then the resnet-18 network will be trained on Cifar-10 dataset.


# Test and generate the heat maps
Run the test.py script, and images together with their associated heat map images will be saved to the directory of "pictures_export"
```python
    $ python test.py
```




