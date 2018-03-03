
#!/usr/bin/python



import mxnet as mx
import numpy as np


def trans_train(data, label):
    '''
    Since the CIFAR-10 dataset images have shapes of (32,32,3), they have to be
    transposed to 'NCWH' in order to be computed through the network
    '''
    data = data.astype(np.float32)
    mean = mx.nd.mean(data)
    data = mx.nd.transpose((data-mean), axes=(2,0,1))

    #  mx.random.seed()
    noise = mx.nd.random.normal(0,1,shape=(3,32,32),dtype=np.float32)
    #  print(noise[0][0][0])

    label.astype(np.uint8)
    return data, label


def trans_test(data, label):
    '''
    Since the CIFAR-10 dataset images have shapes of (32,32,3), they have to be
    transposed to 'NCWH' in order to be computed through the network
    '''
    data = data.astype(np.float32)
    mean = mx.nd.mean(data)
    data = mx.nd.transpose((data-mean), axes=(2,0,1))
    label.astype(np.uint8)
    return data, label



def get_cifar10_iters():
    #  batch_size = config.batch_size
    batch_size = 1

    cifar10_train = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=True,
        transform=trans_train
    )
    cifar10_test = mx.gluon.data.vision.datasets.CIFAR10(
        root='~/.mxnet/datasets/cifar10/',
        train=False,
        transform=trans_test
    )

    train_data = mx.gluon.data.DataLoader(
        cifar10_train,
        batch_size,
        shuffle=True,
        last_batch ='rollover'
    )
    test_data = mx.gluon.data.DataLoader(
        cifar10_test,
        batch_size,
        shuffle=True,
        last_batch='rollover'
    )

    return train_data, test_data


if __name__ == '__main__':
    #  def trans_train(data, label):
    #      # generate noise and add it to images
    #      mx.random.seed(np.random.randint(1,1000))
    #      noise = mx.nd.random.normal(0,1,shape=(3,32,32),dtype=np.float32)
    #      data = data + noise
    #
    #      return data, label
    #
    #  cifar10_train = mx.gluon.data.vision.datasets.CIFAR10(
    #      root='~/.mxnet/datasets/cifar10/',
    #      train=True,
    #      transform=None
    #  )
    #  train_data = mx.gluon.data.DataLoader(
    #      cifar10_train,
    #      batch_size=1,
    #      shuffle=True,
    #      last_batch ='rollover'
    #  )

    noise = mx.nd.random.normal(0,1,shape=(3,32,32),dtype=np.float32)
    print(noise[0][0][0])
    #  for batch, label in train_data:
    #      print(batch[0][0][0])

        #  import numpy as np
        #  mx.random.seed(np.random.randint(1,1000))
        #  noise = mx.nd.random.normal(0,1,shape=(3,32,32),dtype=np.float32)
        #  print(noise[0][0][0])
        #  break
