# MNIST Hand-written Digit Generation with Deep Convolutional GAN (DC-GAN)
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
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
 

# hyperparameters and constants
epoch = 500
batch_size = 128
sample_interval = 10

# input image dimensions
img_shape = (28, 28, 1)

# size of the noise vector, used as input to the generator
noise = 100


# placeholder to store training status
losses = []
accuracies = []
checkpoints = []


# create output directory
OUT_DIR = "./ch3-output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# deep convolution generator
# input: noise input vector (100 values)
# output: 28 x 28 x 1 fake image
def build_generator():

    model = Sequential()

    # reshape input into 7 x 7 x 256 tensor via a fully connected layer
    # convolution needs 3-dimension input
    model.add(Dense(256 * 7 * 7, input_dim = noise))
    model.add(Reshape((7, 7, 256)))

    # 1st transposed convolution layer, from 7 x 7x 256 into 14 x 14 x 128 tensor
    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = 'same'))

    # batch normalization
    model.add(BatchNormalization())

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # 2nd transposed convolution layer, from 14 x 14 x 128 to 14 x 14 x 64 tensor
    model.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = 'same'))

    # batch normalization
    model.add(BatchNormalization())

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # 3rd transposed convolution layer, from 14 x 14 x 64 to 28 x 28 x 1 tensor
    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = 'same'))

    # output layer with tanh activation
    model.add(Activation('tanh'))

    # print model summary
    print('\n=== Generator summary')
    model.summary()

    return model


# deep convolution discriminator
# input: 28 x 28 x 1 
# output: 1 value (real or fake)
def build_discriminator():

    model = Sequential()

    # 1st convolutional layer, from 28 x 28 x 1 into 14 x 14 x 32 tensor
    model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = img_shape, 
        padding = 'same'))

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # 2nd convolutional layer, from 14 x 14 x 32 into 7 x 7 x 64 tensor
    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))

    # batch normalization
    model.add(BatchNormalization())

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # 3rd convolutional layer, from 7 x 7 x 64 tensor into 3 x 3 x 128 tensor
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same'))

    # batch normalization
    model.add(BatchNormalization())

    # leaky ReLU activation
    model.add(LeakyReLU(alpha = 0.01))

    # output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    # print model summary
    print('\n=== Discriminator summary')
    model.summary()

    return model


# GAN definition
# we build generator first and connect discriminator to it
def build_gan(generator, discriminator):

    model = Sequential()

    # combined generator -> discriminator model
    model.add(generator)
    model.add(discriminator)

    return model


# we test generator by showing 16 sample images
def sample_images(itr):
    image_rows = 4
    image_cols = 4

    # sample random noise
    z = np.random.normal(0, 1, (image_rows * image_cols, noise))

    # generate images from random noise
    gen_imgs = generator.predict(z)

    # rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # set image grid
    _, axs = plt.subplots(image_rows, image_cols,
        figsize = (image_rows, image_cols), sharey = True, sharex = True)

    cnt = 0
    for i in range(image_rows):
        for j in range(image_cols):

            # output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1

    # save image files
    path = os.path.join(OUT_DIR, "img-{}".format(itr + 1))
    plt.savefig(path)
    plt.close()


# train discriminator
def disc_train(X_train, real, fake):
    # get a random batch of real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    # generate a batch of fake images
    z = np.random.normal(0, 1, (batch_size, noise))
    gen_imgs = generator.predict(z)

    # train discriminator
    # we train it once with real data and then with fake data
    # note that generator weights are fixed
    # discriminator weights are changing
    # we take the average loss and accuracy between the two
    d_loss_real = discriminator.train_on_batch(imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss, accuracy


# train generator
def gen_train(real):

    # generate a batch of fake images
    z = np.random.normal(0, 1, (batch_size, noise))

    # we want discriminator to say this is real
    # note that discriminator weights are fixed here
    # but generator weights are changing
    g_loss = gan.train_on_batch(z, real)

    return g_loss


# GAN training function
def GAN_train():

    # load MNIST dataset
    # note that we only use the training set
    (X_train, _), (_, _) = mnist.load_data()

    # rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis = 3)

    # labels for real images: all ones
    real = np.ones((batch_size, 1))

    # labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    print('\n\n\nStart GAN training!')

    # start epochs
    for itr in range(epoch):
        d_loss, accuracy = disc_train(X_train, real, fake)
        g_loss = gen_train(real)

        # test generator and save training loss info
        if (itr + 1) % sample_interval == 0:

            # save losses and accuracies for plotting later
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            checkpoints.append(itr + 1)

            # print training progress
            print("Epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (itr + 1, d_loss, 100.0 * accuracy, g_loss))

            # save a sample of generated image
            sample_images(itr)


# plot training losses for discriminator and generator
def loss_plot(losses):
    losses = np.array(losses)
    plt.figure(figsize = (15, 5))

    plt.plot(checkpoints, losses.T[0], label = "Discriminator loss")
    plt.plot(checkpoints, losses.T[1], label = "Generator loss")
    plt.xticks(checkpoints, rotation = 90)

    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    # save the plots at the output directory
    path = os.path.join(OUT_DIR, 'loss-plot.png')
    plt.savefig(path)
    plt.close()


# plot discriminator accuracy
def acc_plot(accuracies):
    accuracies = np.array(accuracies)
    plt.figure(figsize=(15, 5))
    plt.plot(checkpoints, accuracies, label = "Discriminator accuracy")

    plt.xticks(checkpoints, rotation=90)
    plt.yticks(range(0, 100, 5))

    plt.title("Discriminator Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # save the plots at the output directory
    path = os.path.join(OUT_DIR, 'acc-plot.png')
    plt.savefig(path)
    plt.close()


#################
#  MAIN ROUTINE #
#################

# build and compile discriminator
discriminator = build_discriminator()
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam',
    metrics = ['accuracy'])

# build generator
# note that we do not compile generator
generator = build_generator()

# keep discriminator’s parameters constant for generator training
discriminator.trainable = False

# build and compile GAN model with fixed discriminator to train generator
gan = build_gan(generator, discriminator)
gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')

# train DC-GAN for the specified number of epochs
# then draw loss and accuracy plots
GAN_train()
loss_plot(losses)
acc_plot(accuracies)

