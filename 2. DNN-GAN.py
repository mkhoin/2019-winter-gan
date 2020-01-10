# MNIST Hand-written Digit Generation with Deep Neural Network GAN (DNN-GAN)
# Dataset: MNIST
# January 9, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam


# global constants and hyper-parameters
epoch = 10000
batch_size = 100
sample_interval = 100
noise = 100
img_shape = (28, 28, 1)


# create output directory
OUT_DIR = "./ch2-output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# build generator using a simple DNN
def build_generator():

    model = Sequential()

    # fully connected layer
    model.add(Dense(128, input_dim = noise))

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # output layer with tanh activation
    model.add(Dense(28 * 28 * 1, activation = 'tanh'))

    # reshape generator output to image dimensions
    # 784 output neurous used
    model.add(Reshape(img_shape))
    print('\n=== Generator summary')
    model.summary()

    return model


# build discriminator using a simple DNN
def build_discriminator():

    model = Sequential()

    # flatten the input image
    model.add(Flatten(input_shape = img_shape))
    
    # fully connected layer
    model.add(Dense(128))

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # output layer with sigmoid activation
    # it produces a probability value
    model.add(Dense(1, activation = 'sigmoid'))
    print('\n=== Discriminator summary')
    model.summary()

    return model


# build GAN
# we connect the output of generator 
# to the input of discriminator
def build_gan():

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    print('\n=== GAN summary')
    model.summary()

    return model


# read MNIST dataset
def read_dataset():
    # load the MNIST dataset
    # note that we only need train data input
    (X_train, _), (_, _) = mnist.load_data()
    print('\n=== Train input set shape:', X_train.shape)

    # rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis = 3)
    print('=== Train input set after reshaping:', X_train.shape)

    return X_train


# GAN training routine
def train_GAN():

    # labels for real images: all ones
    # why double parenthesis?
    real = np.ones((batch_size, 1))
    print('\n=== Real label shape:', real.shape)

    # labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))
    print('=== Fake label shape:', fake.shape)

    # training loop
    for itr in range(epoch):

        ########################
        #  train discriminator #
        ########################

        # get a random batch of real images
        # we choose some number of data out of 60K train set
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        if itr == 0:
            print('=== Random training sample set shape:', imgs.shape)

        # generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, noise))
        if itr == 0:
            print('=== Genertor input shape:', z.shape)
        fake_imgs = generator.predict(z)
        if itr == 0:
            print('\n=== Fake image shape:', fake_imgs.shape)

        # train discriminator
        # during discriminator training, generator weights are untouched
        # train_on_batch() returns two numbers: loss and accuracy
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)

        # we take the average between the two
        d_loss, d_acc = 0.5 * np.add(d_loss_real, d_loss_fake)

        ####################
        #  train generator #
        ####################

        # generate a batch of fake images
        # same as in discriminator
        z = np.random.normal(0, 1, (batch_size, noise))

        # note we train the entire gan but fix discriminator weights
        # we fool discrminator that generator is producing real images
        # generator does not generate accuracy
        discriminator.trainable = False
        g_loss = gan.train_on_batch(z, real)

        ##########################
        #  print training status #
        ##########################

        if (itr + 1) % sample_interval == 0:

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (itr + 1, d_loss, 100.0 * d_acc, g_loss))

            # Output a sample of generated image
            sample_images(itr)


def sample_images(itr):

    row = col = 4

    # Sample random noise
    z = np.random.normal(0, 1, (row * col, noise))

    # generate 16 fake images from random noise
    fake_imgs = generator.predict(z)

    # rescale image pixel values to [0, 1]
    fake_imgs = 0.5 * fake_imgs + 0.5

    # set image grid
    _, axs = plt.subplots(row, col, figsize = (row, col),
        sharey = True, sharex = True)

    cnt = 0
    for i in range(row):
        for j in range(col):

            # output a grid of images
            axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1

    path = os.path.join(OUT_DIR, "img-{}".format(itr + 1))
    plt.savefig(path)
    plt.close()


# main build and compile module
# read MNIST dataset, only thw training input set
X_train = read_dataset()


#################
#  MAIN ROUTINE #
#################

# build discriminator
discriminator = build_discriminator()
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'Adam',
                      metrics=['accuracy'])

# we do not compile generator separately
generator = build_generator()

# keep discriminator’s parameters constant for generator training
discriminator.trainable = False


########################
#  BUILD AND TRAIN GAN #
########################

# build and compile GAN with fixed discriminator to train generator
gan = build_gan()
gan.compile(loss = 'binary_crossentropy', optimizer = 'Adam')


# train GAN for the specified number of epochs
train_GAN()
