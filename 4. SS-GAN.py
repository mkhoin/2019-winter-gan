# MNIST Hand-written Digit Generation with Semi-Supervied GAN (SS-GAN)
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
from keras.layers import Activation, BatchNormalization, Concatenate, Dense
from keras.layers import Dropout, Flatten, Input, Lambda, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


# input image dimensions
img_shape = (28, 28, 1)

# size of the noise vector, used as input to the Generator
noise = 100

# number of classes in the dataset
num_classes = 10

# set hyperparameters
epochs = 500
batch_size = 100
sample_interval = 10

# number of labeled examples to use (rest will be used as unlabeled)
num_labeled = 100


# create output directory
OUT_DIR = "./ch4-output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# get a random batch of labeled images and their labels
def batch_labeled():
    idx = np.random.randint(0, num_labeled, batch_size)
    imgs = X_train[idx]
    labels = Y_train[idx]

    return imgs, labels


# get a random batch of unlabeled images
def batch_unlabeled():
    idx = np.random.randint(num_labeled, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    return imgs


# get the labeled training set
def training_set():
    x = X_train[range(num_labeled)]
    y = Y_train[range(num_labeled)]

    return x, y


# get the test set
def test_set():
    return X_test, Y_test


# build CNN-based generator
# input: one dimensional noise array 
# output: 28 x 28 x 1 fake image
def build_generator(noise):

    model = Sequential()

    # reshape noise input into 7x7x256 tensor via dense layer
    model.add(Dense(256 * 7 * 7, input_dim = noise))
    model.add(Reshape((7, 7, 256)))

    # 1st transposed convolution block
    # dimension becomes 14 x 14 x 128
    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    # 2nd transposed convolution block
    # dimension becomes 14 x 14 x 64
    # image size stays the same
    model.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    # 3rd transposed convolution block
    # dimension becomes 28 x 28 x 1
    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = 'same'))

    # output layer with tanh activation
    model.add(Activation('tanh'))

    # print the summary of generator
    print('\n=== Generator summary')
    model.summary()

    return model


# build CNN-based discriminator (common part)
# input: 28 x 28 x 1 image (real or fake) 
# output: 10 dense neuron outputs
def build_disc_common(img_shape):

    model = Sequential()

    # 1st convolutional block
    # dimension becomes 14 x 14 x 32
    model.add(Conv2D(32, kernel_size = 3, strides = 2,
        input_shape = img_shape, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.01))

    # 2nd convolutional block
    # dimension becomes 7 x 7 x 64
    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    # 3rd convolutional block
    # dimension becomes 3 x 3 x 128
    model.add(Conv2D(128, kernel_size=3, strides = 2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dropout(0.5))

    # flatten the tensor
    model.add(Flatten())

    # fully connected layer with num_classes neurons
    model.add(Dense(num_classes))

    # print the summary of discriminator
    print('\n=== Discriminator (common) summary')
    model.summary()

    return model


# add softmax activation at the end of discriminator
# giving predicted probability distribution over the real classes
# input: 10 values
# output: 10 softmax probability values
def build_disc_super(disc_common):

    model = Sequential()
    model.add(disc_common)
    model.add(Activation('softmax'))

    return model


# add sigmoid activation at the end of discriminator
# to decide real vs. fake 
# input: 10 values
# output: single probability value 
def build_disc_unsuper(disc_common):

    model = Sequential()
    model.add(disc_common)
    model.add(Dense(1, activation = 'sigmoid'))

    return model


# build GAN
# we connect the output of generator 
# to the input of discriminator
def build_gan(generator, discriminator):

    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    print('\n=== GAN summary')
    model.summary()

    return model


#############
# BUILD GAN #
#############

# build the common discriminator
# these layers are shared between supervised and unsupervised training
disc_common = build_disc_common(img_shape)


# build & compile the discriminator for supervised training
disc_super = build_disc_super(disc_common)
disc_super.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',
    metrics = ['accuracy'])


# build & compile the discriminator for unsupervised training
disc_unsuper = build_disc_unsuper(disc_common)
disc_unsuper.compile(optimizer = 'Adam', loss = 'binary_crossentropy')


# build generator
generator = build_generator(noise)


# keep discriminator’s parameters constant for generator training
disc_unsuper.trainable = False


# build & compile GAN model with fixed discriminator to train generator
# we are using the discriminator with unsupervised output
gan = build_gan(generator, disc_unsuper)
gan.compile(optimizer = 'adam', loss = 'binary_crossentropy')


#############
# TRAIN GAN #
#############


# train discriminator for unsupervised version
def train_disc():

    # labels for real images: all ones
    real = np.ones((batch_size, 1))

    # labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))
    
    # get labeled examples
    imgs, labels = batch_labeled()
    labels = to_categorical(labels, num_classes)

    # get unlabeled examples
    imgs_unlabeled = batch_unlabeled()

    # generate a batch of fake images
    z = np.random.normal(0, 1, (batch_size, noise))
    gen_imgs = generator.predict(z)

    # supervised train on real labeled examples
    d_loss_super, _ = disc_super.train_on_batch(imgs, labels)

    # unsupervised train on real unlabeled examples
    d_loss_real = disc_unsuper.train_on_batch(imgs_unlabeled, real)

    # unsupervised train on fake examples
    d_loss_fake = disc_unsuper.train_on_batch(gen_imgs, fake)

    # computing loss by taking the average
    d_loss_unsuper = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss_super, d_loss_unsuper


# train generator
def train_gen():

    # generate a batch of fake images
    z = np.random.normal(0, 1, (batch_size, noise))

    # train generator
    g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))

    return g_loss


# main training routine
def train():

    print('\n=== GAN TRAINING BEGINS')
    for itr in range(epochs):
        d_loss_s, d_loss_u = train_disc()
        g_loss = train_gen()

        if (itr + 1) % sample_interval == 0:

            # output training progress
            print("%d [D loss super: %.4f], [D loss unsuper: %.4f], [G loss: %f]" 
                % (itr + 1, d_loss_s, d_loss_u, g_loss))
            sample_images(itr)                


# select 16 sample images to test generator
# and display/save the results
def sample_images(itr):

    row = col = 4

    # sample random noise
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


##################
# MAIN EXECUTION #
##################

# read and pre-process MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
Y_train = Y_train.reshape(60000, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
Y_test = Y_test.reshape(10000, 1)

X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


# train GAN for the specified number of epochs
train()


#################################
# SEMI-SUPERVISED DISCRIMINATOR #
#################################

# evaluating semi-supervised trained discriminator
# compute classification accuracy on the training set
print('\n=== SEMI-SUPERVISED DISCRMINATOR EVALUATION')
x, y = training_set()
y = to_categorical(y, num_classes = num_classes)
_, accuracy = disc_super.evaluate(x, y, verbose = 0)
print("Training accuracy: %.2f%%" % (100 * accuracy))


# compute classification accuracy on the test set
x, y = test_set()
y = to_categorical(y, num_classes = num_classes)
_, accuracy = disc_super.evaluate(x, y, verbose = 0)
print("Test accuracy: %.2f%%" % (100 * accuracy))


##################################
# FULLY SUPERVISED DISCRIMINATOR #
##################################

# use the same network architecture as the SS-GAN discriminator
fully_super = build_disc_super(build_disc_common(img_shape))
fully_super.compile(loss = 'categorical_crossentropy',
    metrics = ['accuracy'], optimizer = 'Adam')


# process train set
imgs, labels = training_set()
labels = to_categorical(labels, num_classes = num_classes)


# train the classifier using keras fitting function, not GAN
training = fully_super.fit(x = imgs, y = labels, batch_size = batch_size,
    epochs = epochs, verbose = 0)


# compute classification accuracy on the training set
print('\n=== FULLY-SUPERVISED DISCRMINATOR EVALUATION')
x, y = training_set()
y = to_categorical(y, num_classes = num_classes)
_, accuracy = fully_super.evaluate(x, y, verbose = 0)
print("Training accuracy: %.2f%%" % (100 * accuracy))


# compute classification accuracy on the test set
x, y = test_set()
y = to_categorical(y, num_classes = num_classes)
_, accuracy = fully_super.evaluate(x, y, verbose = 0)
print("Test accuracy: %.2f%%" % (100 * accuracy))
