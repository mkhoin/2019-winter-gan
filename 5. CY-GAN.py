# Turning Paintings into Photos using Cycle-GAN (CY-GAN)
# Dataset: Monet Paintings and Photos
# January 9, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from glob import glob
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Add, Conv2DTranspose, ZeroPadding2D, LeakyReLU
from keras.optimizers import Adam
from imageio import imread
from skimage.transform import resize


# hyper-parameters and constants
batch_size = 2
epochs = 1
data_dir = "./dataset/monet2photo/"
input_shape = (128, 128, 3)
residual_blocks = 6
hidden_layers = 3
my_opt = Adam(0.002, 0.5)


# execution mode: train or test
mode = 'train'


# create directories
OUT_DIR = "./ch5-output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# residual block used in the two generators
def residual_block(x):
    res = Conv2D(128, kernel_size = 3, strides = 1, padding = "same")(x)
    res = BatchNormalization(axis = 3, momentum=0.9, epsilon = 1e-5)(res)
    res = Activation('relu')(res)

    res = Conv2D(128, kernel_size = 3, strides = 1, padding = "same")(res)
    res = BatchNormalization(axis = 3, momentum = 0.9, epsilon = 1e-5)(res)

    return Add()([res, x])


# generator definition
# it has an auto-encoder shape
def build_generator():
    input_layer = Input(shape = input_shape)

    # 1st convolution block
    x = Conv2D(32, kernel_size = 7, strides = 1, padding = "same")(input_layer)
    x = BatchNormalization(axis = 1)(x)
    x = Activation("relu")(x)

    # 2nd convolution block
    x = Conv2D(64, kernel_size = 3, strides = 2, padding = "same")(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation("relu")(x)

    # 3rd convolution block
    x = Conv2D(128, kernel_size = 3, strides = 2, padding = "same")(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation("relu")(x)

    # residual blocks
    for _ in range(residual_blocks):
        x = residual_block(x)

    # 1st upsampling block
    x = Conv2DTranspose(64, kernel_size = 3, strides = 2, padding='same', use_bias = False)(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation("relu")(x)

    # 2nd upsampling block
    x = Conv2DTranspose(32, kernel_size = 3, strides = 2, padding='same', use_bias = False)(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation("relu")(x)

    # final convolution layer
    x = Conv2D(3, kernel_size = 7, strides = 1, padding = "same")(x)
    output = Activation('tanh')(x)

    model = Model(inputs = [input_layer], outputs = [output])
    return model


# discriminator definition
# it consists of convolution blocks
def build_discriminator():
    input_layer = Input(shape = input_shape)
    x = ZeroPadding2D(padding = (1, 1))(input_layer)

    # 1st convolutional block
    x = Conv2D(64, kernel_size = 4, strides = 2, padding = "valid")(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = ZeroPadding2D(padding = (1, 1))(x)

    # 3 hidden convolution blocks
    for i in range(1, hidden_layers + 1):
        x = Conv2D(2 ** i * 64, kernel_size = 4, strides = 2, padding = "valid")(x)
        x = BatchNormalization(axis = 1)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = ZeroPadding2D(padding = (1, 1))(x)

    # final convolution layer
    output = Conv2D(1, kernel_size = 4, strides = 1, activation = "sigmoid")(x)

    model = Model(inputs = [input_layer], outputs = [output])
    return model


# loading image files using glob package
def load_images():
    imagesA = glob(data_dir + 'testA/*.*')
    imagesB = glob(data_dir + 'testB/*.*')

    allImagesA = []
    allImagesB = []

    print('\nImages in the dataset:')
    print('   Test A (paintings):', len(imagesA))
    print('   Test B (photos):', len(imagesB))

    for index, filename in enumerate(imagesA):
        imgA = imread(filename, pilmode='RGB')
        imgB = imread(imagesB[index], pilmode='RGB')

        imgA = resize(imgA, (128, 128))
        imgB = resize(imgB, (128, 128))

        if np.random.random() > 0.5:
            imgA = np.fliplr(imgA)
            imgB = np.fliplr(imgB)

        allImagesA.append(imgA)
        allImagesB.append(imgB)

    # Normalize images
    allImagesA = np.array(allImagesA) / 127.5 - 1.
    allImagesB = np.array(allImagesB) / 127.5 - 1.

    print('   Paintings used:', len(allImagesA))
    print('   Photos used:', len(allImagesB))

    return allImagesA, allImagesB


# build two discriminators
# A is painting, B is photo
# note that discriminators need compilation
def build_discriminators():

    # discriminator A summary
    # input: 128 x 128 x 3 image
    # output: 7 x 7 x 1 values
    discriminatorA = build_discriminator()    
    print('\n=== Discriminator A summary')
    discriminatorA.summary()

    # discriminator B summary
    # input: 128 x 128 x 3 image
    # output: 7 x 7 x 1 values
    discriminatorB = build_discriminator()    
    print('\n=== Discriminator B summary')
    discriminatorB.summary()

    discriminatorA.compile(loss = 'mse', optimizer = my_opt, metrics = ['accuracy'])
    discriminatorB.compile(loss = 'mse', optimizer = my_opt, metrics = ['accuracy'])    
    
    return discriminatorA, discriminatorB


# build two generators
# A is painting, B is photo
# note that generators do not need compilation
def build_generators():

    # generator A to B 
    # input: 128 x 128 x 3 image
    # output: 128 x 128 x 3 image
    generatorAtoB = build_generator()    
    print('\n=== Generator A to B summary')
    generatorAtoB.summary()

    # generator B to A 
    # input: 128 x 128 x 3 image
    # output: 128 x 128 x 3 image
    generatorBtoA = build_generator()
    print('\n=== Generator B to A summary')
    generatorBtoA.summary()

    return generatorAtoB, generatorBtoA


# build cycle-GAN
# A is painting, B is photo
# note that CGAN needs compilation
# input: two images (one painting, one photo)
# output: two generated images, two real/fake answers 
# additional output: error-related for back-propagation
def build_GAN(discriminatorA, discriminatorB, generatorAtoB, generatorBtoA):

    # input layers
    inputA = Input(shape = input_shape)
    inputB = Input(shape = input_shape)

    # generate images using both generators
    genB = generatorAtoB(inputA)
    genA = generatorBtoA(inputB)

    # reconstruct images back to original 
    reconA = generatorBtoA(genB)
    reconB = generatorAtoB(genA)

    # additional outputs
    genA_ID = generatorBtoA(inputA)
    genB_ID = generatorAtoB(inputB)
    probs_A = discriminatorA(genA)
    probs_B = discriminatorB(genB)

    # make both discriminators non-trainable
    discriminatorA.trainable = False
    discriminatorB.trainable = False

    # build and compile cycle-GAN
    CGAN = Model(inputs = [inputA, inputB], 
        outputs = [probs_A, probs_B, reconA, reconB, genA_ID, genB_ID])

    CGAN.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
        loss_weights = [1, 1, 10.0, 10.0, 1.0, 1.0],
        optimizer = my_opt)

    print('\n=== Cycle-GAN summary')
    CGAN.summary()

    return CGAN   


# pick the next batch of random images
# to test GAN and save the resulting images
def load_test_batch():
    imagesA = glob(data_dir + 'testA/*.*')
    imagesB = glob(data_dir + 'testB/*.*')

    imagesA = np.random.choice(imagesA, batch_size)
    imagesB = np.random.choice(imagesB, batch_size)

    allA = []
    allB = []

    for i in range(len(imagesA)):
        # load and resize images
        imgA = resize(imread(imagesA[i], pilmode='RGB').astype(np.float32), (128, 128))
        imgB = resize(imread(imagesB[i], pilmode='RGB').astype(np.float32), (128, 128))

        allA.append(imgA)
        allB.append(imgB)

    # scaling the raw data
    return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0


# show 3 images per sample: original, generated, and reconstructed
# the first set is paintings, the secone set photos
def save_images(oriA, genB, reconA, oriB, genA, reconB, path):
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow((oriA * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow((genB * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow((reconA * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Reconstructed")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow((oriB * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow((genA * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow((reconB * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)


# test CGAN on the first batch
def test_CGAN(epoch, index):
    # Get two sample sets
    batchA, batchB = load_test_batch()

    # Generate images
    genB = generatorAtoB.predict(batchA)
    genA = generatorBtoA.predict(batchB)

    # Get reconstructed images
    reconA = generatorBtoA.predict(genB)
    reconB = generatorAtoB.predict(genA)

    # save one image file per epoch per iteration
    save_images(batchA[0], genB[0], reconA[0], batchB[0], genA[0], reconB[0],
        path = os.path.join(OUT_DIR, "gen_{}_{}".format(epoch + 1, index + 1)))


# train cycle-GAN
if mode == 'train':
    print('\n\n\nCGAN training starts!')
    # load the dataset
    imagesA, imagesB = load_images()

    # build the necessary networks
    discriminatorA, discriminatorB = build_discriminators()
    generatorAtoB, generatorBtoA = build_generators()
    CGAN = build_GAN(discriminatorA, discriminatorB, generatorAtoB, generatorBtoA)

    # real and fake labels for discriminators
    real_labels = np.ones((batch_size, 7, 7, 1))
    fake_labels = np.zeros((batch_size, 7, 7, 1))

    # epoch training starts
    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))

        num_batches = int(min(imagesA.shape[0], imagesB.shape[0]) / batch_size / 3)
        print("   Number of batches: {}".format(num_batches))

        for index in range(num_batches):
            print("      Batch {}".format(index))

            # obtain the next batch of images
            batchA = imagesA[index * batch_size:(index + 1) * batch_size]
            batchB = imagesB[index * batch_size:(index + 1) * batch_size]

            # mpa images to opposite domain
            genB = generatorAtoB.predict(batchA)
            genA = generatorBtoA.predict(batchB)

            # train discriminator A on real and fake images
            dA_loss_r = discriminatorA.train_on_batch(batchA, real_labels)
            dA_loss_f = discriminatorA.train_on_batch(genA, fake_labels)

            # train discriminator B on ral and fake images
            dB_loss_r = discriminatorB.train_on_batch(batchB, real_labels)
            dB_loss_f = discriminatorB.train_on_batch(genB, fake_labels)

            # calculate the total discriminator loss
            d_loss = 0.5 * np.add(0.5 * np.add(dA_loss_r, dA_loss_f), 
                0.5 * np.add(dB_loss_r, dB_loss_f))

            print("         d_loss:{}".format(d_loss))

            # train generators and calculate loss
            g_loss = CGAN.train_on_batch([batchA, batchB],
                [real_labels, real_labels, batchA, batchB, batchA, batchB])

            print("         g_loss:{}".format(g_loss))

            test_CGAN(epoch, index)


    # save models for prediction 
    generatorAtoB.save_weights(os.path.join(OUT_DIR, "generatorAToB.h5"))
    generatorBtoA.save_weights(os.path.join(OUT_DIR, "generatorBToA.h5"))


# use cycle-GAN for prediction
if mode == 'test':
    print('\n\n\nCGAN prediction starts!')

    # build generator networks
    generatorAtoB = build_generator()
    generatorBtoA = build_generator()

    generatorAtoB.load_weights(os.path.join(OUT_DIR, "generatorAToB.h5"))
    generatorBtoA.load_weights(os.path.join(OUT_DIR, "generatorBToA.h5"))

    # get a batch of test data
    batchA, batchB = load_test_batch()

    # generate fake images
    genB = generatorAtoB.predict(batchA)
    genA = generatorBtoA.predict(batchB)

    reconA = generatorBtoA.predict(genB)
    reconB = generatorAtoB.predict(genA)

    # save images
    for i in range(len(genA)):
        save_images(batchA[i], genB[i], reconA[i], batchB[i], genA[i], reconB[i],
            path = os.path.join(OUT_DIR, "test_{}".format(i)))

