# Spread Matching with Basic GAN (B-GAN)
# Dataset: randon normal distribution
# January 9, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam


# create directories
OUT_DIR = "./ch1-output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# global constants and hyper-parameters
NUM_DATA = 100
MY_BATCH = 2
MY_STAGE = 10
MY_EPOCH = 100
MY_TEST = 20

# number of neurons
D_SIZE = 50
G_SIZE = 50

# number of times trained in each epoch
D_LEARN = 1
G_LEARN = 5

# additional definitions
MY_MU = 10
MY_SIGMA = 0.25
MY_ADAM = Adam(lr = 0.0002, beta_1 = 0.9, beta_2 = 0.999)


# we use the same compilation setting for all models
def model_compile(model):
    return model.compile(loss = 'binary_crossentropy', optimizer = MY_ADAM, 
            metrics = ['accuracy'])


####################
# DATABASE SETTING #
####################


# random samples with mean = mu and stddev = sigma
def real_sample():
    return np.random.normal(MY_MU, MY_SIGMA, (MY_BATCH, NUM_DATA))


# random samples from a uniform distribution over [0, 1).
def in_sample():
    return np.random.rand(MY_BATCH, NUM_DATA)


##################
# MODEL BUILDING #
##################


# DNN discriminator definition
def build_D():
    D = models.Sequential()
    D.add(Dense(D_SIZE, activation = 'relu', input_shape = (NUM_DATA,)))
    D.add(Dense(D_SIZE, activation = 'relu'))
    D.add(Dense(D_SIZE, activation = 'relu'))
    D.add(Dense(1, activation = 'sigmoid'))
    model_compile(D)

    print('\n== DISCRIMINATOR MODEL DETAILS ==')
    D.summary()
    return D


# CNN generator definition
def build_G():
    G = models.Sequential()
    G.add(Reshape((NUM_DATA, 1), input_shape = (NUM_DATA,)))
    G.add(Conv1D(G_SIZE, 1, activation = 'relu'))
    G.add(Conv1D(G_SIZE, 1, activation = 'sigmoid'))
    G.add(Conv1D(1, 1))
    G.add(Flatten())
    model_compile(G)

    print('\n== GENERTOR MODEL DETAILS ==')
    G.summary()
    return G


# GAN definition
def build_GAN(dis, gen):
    GAN = models.Sequential()
    GAN.add(gen)
    GAN.add(dis)
    dis.trainable = False
    model_compile(GAN)

    print('\n== GAN MODEL DETAILS ==')
    GAN.summary()
    return GAN


##################
# MODEL TRAINING #
##################


# run epochs
# in each epoch, we train generator more than discriminator 
def train_epochs(gan, dis, gen):
    for e in range(MY_EPOCH):

        # discriminator training
        # we train using a real data first then a fake next
        for i in range(D_LEARN):
            real = real_sample()
            Z = in_sample()

            fake = gen.predict(Z)
            dis.trainable = True
            X = np.concatenate([real, fake], axis = 0)
            y = np.array([1] * MY_BATCH + [0] * MY_BATCH)
            dis.train_on_batch(X, y)


        # generator training
        # each time we train using an input that produces fake
        # goal is to fool discriminator
        for i in range(G_LEARN):
            Z = in_sample()
            dis.trainable = False
            y = np.array([1] * MY_BATCH)
            gan.train_on_batch(Z, y)


####################
# MODEL EVALUATION #
####################

# we test generator
# we compare real and fake and want them to be close
def test_and_show(gen):

    Z = np.random.rand(MY_TEST, NUM_DATA)
    fake = gen.predict(Z)
    real = np.random.normal(MY_MU, MY_SIGMA, (MY_TEST, NUM_DATA))
    
    plt.hist(real.reshape(-1), histtype = 'step', label = 'Real')
    plt.hist(fake.reshape(-1), histtype = 'step', label = 'Generated')
    plt.hist(Z.reshape(-1), histtype = 'step', label = 'Input')
    plt.legend(loc = 0)

    print('        Real: mean = {:.2f}'.format(np.mean(real)), 
            ', std-dev = {:.2f}'.format(np.std(real)))
    
    print('        Fake: mean = {:.2f}'.format(np.mean(fake)), 
            ', std-dev = {:.2f}'.format(np.std(fake)))


# train and test GAN
def run(gan, dis, gen):

    for i in range(MY_STAGE):
        print('\nStage', i, '(Epoch: {})'.format(i * MY_EPOCH))

        train_epochs(gan, dis, gen)
        test_and_show(gen)

        path = os.path.join(OUT_DIR, "img-{}".format(i))
        plt.savefig(path)
        plt.close()


# build and run GAN 
dis = build_D()
gen = build_G()
GAN = build_GAN(dis, gen)
run(GAN, dis, gen)
