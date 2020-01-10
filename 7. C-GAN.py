# Face Aging Using Conditional GAN (C-GAN)
# Dataset: Face Shots
# January 9, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense
from keras.layers import BatchNormalization

from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.layers import Activation, UpSampling2D, Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing import image
from scipy.io import loadmat
from keras.models import Sequential


# directories
DB_DIR = "./dataset/face/"
OUT_DIR = "./ch7-output/"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# hyper-parameters and constants
epochs = 2
batch_size = 100
noise = 100
num_class = 6
num_data = 1000
DB_NAME = '1000-data.mat'
image_shape = (64, 64, 3)
fr_image_shape = (192, 192, 3)


# program control option 1: on-off-off
# program control option 2: on-on-off then off-off-on
TRAIN_GAN = True
TRAIN_ENCODER = False
TRAIN_GAN_WITH_FR = False


# encoder network
# input: 64 x 64 x 3 image
# output: 100 numbers (= noise vector)
def build_encoder():
    input_layer = Input(shape = image_shape)

    # 1st convolutional block
    enc = Conv2D(32, kernel_size = 5, strides = 2, padding = 'same')(input_layer)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)

    # 2nd convolutional block
    enc = Conv2D(64, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)

    # 3rd convolutional block
    enc = Conv2D(128, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)

    # 4th convolutional block
    enc = Conv2D(256, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)

    # flatten layer
    enc = Flatten()(enc)

    # 1st fully connected layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)

    # 2nd fully connected layer
    enc = Dense(noise)(enc)

    # create a model
    model = Model(inputs = [input_layer], outputs = [enc])
    print('\n=== Encoder summary')
    model.summary()

    return model


# CNN generator definition
# input: 106 values
# output: 64 x 64 x 3 RGB image
def build_generator():
    input_noise = Input(shape = (noise,))
    input_label = Input(shape = (num_class,))

    # we concatenate input noise and class dimensions
    x = concatenate([input_noise, input_label])

    # 1st fully connected layer
    x = Dense(2048, input_dim = noise + num_class)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    # 2nd fully connected layer
    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    # reshaping to enter convolution layer
    x = Reshape((8, 8, 256))(x)

    # 1st upscaling and convolution block
    # dimension becomes 16 x 16 x 128
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(128, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # 2nd upscaling and convolution block
    # dimension becomes 32 x 32 x 64
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(64, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # 3rd upscaling and convolution block
    # dimension becomes 64 x 64 x 3
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(3, kernel_size = 5, padding = 'same')(x)
    x = Activation('tanh')(x)

    # create a model
    model = Model(inputs=[input_noise, input_label], outputs=[x])
    print('\n=== Generator summary')
    model.summary()

    return model


# custom keras layer for dimension expansion
def expand_label(x):
    x = K.expand_dims(x, axis = 1)
    x = K.expand_dims(x, axis = 1)
    x = K.tile(x, [1, 32, 32, 1])

    return x


# CNN discriminator definition
# input: 64 x 64 x 3 image and 6 additional values (= age class)
# output: single probability value (fake vs. real)
def build_discriminator():
    image_input = Input(shape = image_shape)
    label_input = Input(shape = (num_class,))

    # 1st convolution block
    # dimension becomes 32 x 32 x 64
    x = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same')(image_input)
    x = LeakyReLU(alpha = 0.2)(x)

    # expand label dimension 
    # dimension becomes 32 x 32 x 6
    label_input1 = Lambda(expand_label)(label_input)

    # concatenate two inputs
    # dimension becomes 32 x 32 x 70
    x = concatenate([x, label_input1], axis = 3)

    # 2nd convolution block
    # dimension becomes 16 x 16 x 128
    x = Conv2D(128, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # 3rd convolution block
    # dimension becomes 8 x 8 x 256
    x = Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # 4th convolution block
    # dimension becomes 4 x 4 x 512
    x = Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # flattened to get 8192 neurons
    x = Flatten()(x)

    # final output is single value
    x = Dense(1, activation = 'sigmoid')(x)

    # print the summary of discriminator
    model = Model(inputs=[image_input, label_input], outputs = [x])
    print('\n=== Discriminator summary')
    model.summary()

    return model


# taken: year the photo was taken
# dob: date of birth in ordinal format
# birth: birthday in yyyy-mm-dd format
def calculate_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


# load image data
# we down select 1000 images out of 62328 
def load_data():

    # load the matlab file
    # meta contains 4 dimensional dictionary
    dataset = 'wiki'
    meta = loadmat(os.path.join(DB_DIR, DB_NAME))
    full_path = meta[dataset][0, 0]["full_path"][0]
    print('\nWe use', len(full_path), 'images.')

    # List of date-of-birth and year-taken data
    dob = meta[dataset][0, 0]["dob"][0]
    photo_taken = meta[dataset][0, 0]["photo_taken"][0]

    # Calculate age for all data
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # create a list of tuples containing image path and age
    images = []
    ages = []

    for index, image_path in enumerate(full_path):
        images.append(image_path[0])
        ages.append(age[index])

    # Return a list of all images and respective age
    return images, ages


# categorize all ages into 6 groups
def age_to_category(ages):
    category = []

    for age in ages:
        if 0 < age <= 18:
            age_category = 0
        elif 18 < age <= 29:
            age_category = 1
        elif 29 < age <= 39:
            age_category = 2
        elif 39 < age <= 49:
            age_category = 3
        elif 49 < age <= 59:
            age_category = 4
        elif age >= 60:
            age_category = 5

        category.append(age_category)

    return category


# load, resize, and convert images in the dataset
def load_image(image_paths, image_shape):
    images = None

    print("Image loading began.")
    for i, path in enumerate(image_paths):
        # load and resize images using keras image package
        loaded_image = image.load_img(os.path.join(DB_DIR, path), 
            target_size = image_shape)

        # convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # add another dimension (add batch dimension)
        # dimension becomes (1, 64, 64, 3)
        loaded_image = np.expand_dims(loaded_image, axis = 0)

        # concatenate all images into one tensor
        # dimension becomes (1000, 64, 64, 3)
        if images is None:
            images = loaded_image
        else:
            images = np.concatenate([images, loaded_image], axis = 0)

    print("Image loading done.")

    return images


# euclidean distance loss
# y_true: tensor
# y_pred: tensor of the same shape as y_true
# return: float
def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# save the given color image to the output directory
def save_rgb_img(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow((img * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)   
    plt.close()


#############
# BUILD GAN #
#############

# build and compile discriminator
discriminator = build_discriminator()
discriminator.compile(loss = ['binary_crossentropy'], optimizer = 'adam')
discriminator.trainable = False

# build and compile generator 
generator = build_generator()
generator.compile(loss = ['binary_crossentropy'], optimizer = 'adam')

# build and compile GAN
input_noise = Input(shape = (100,))
input_label = Input(shape = (6,))
recon_image = generator([input_noise, input_label])
valid = discriminator([recon_image, input_label])

# setting up inputs and output
GAN_model = Model(inputs = [input_noise, input_label], outputs = [valid])
GAN_model.compile(loss = ['binary_crossentropy'], optimizer = 'adam')

print('\n=== GAN summary')
GAN_model.summary()


################
# LOAD DATASET #
################

# obtain image path and age category data
paths, ages = load_data()
age_cat = age_to_category(ages)

# reshaping 1000 category data to (1000, 1)
final_age_cat = np.reshape(np.array(age_cat), [num_data, 1])

# one-hot encoded category data
cats = to_categorical(final_age_cat, num_class)

# load image data
images = load_image(paths, (image_shape[0], image_shape[1]))


#####################
# TRAIN INITIAL GAN #
#####################

# train discriminator
def train_disc(img_batch, y_batch, z_noise):
    # generate fake images
    fakes = generator.predict_on_batch([z_noise, y_batch])

    # implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype = np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype = np.float32) * 0.1
    
    # train discriminator using real and fake images
    # we need both inputs: image and its age category
    d_loss_real = discriminator.train_on_batch([img_batch, y_batch], real_labels)
    d_loss_fake = discriminator.train_on_batch([fakes, y_batch], fake_labels)

    # calculate the average between real and fake loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss


# train generator
def train_gen():

    # generate random noise data (batch x 100 values)
    z_noise = np.random.normal(0, 1, size = (batch_size, noise))

    # generate random category data (batch x 6 values)
    # then one-hot encode it
    random_labels = np.random.randint(0, num_class, batch_size).reshape(-1, 1)
    random_labels = to_categorical(random_labels, 6)

    # train generator with real label
    g_loss = GAN_model.train_on_batch([z_noise, random_labels], [1] * batch_size)

    return g_loss


# test the generator with the first five images
# then save the resulting image files
def test_gen(epoch):

    # fetch the first image batch
    img_batch = images[0:batch_size]
    img_batch = img_batch / 127.5 - 1.0
    img_batch = img_batch.astype(np.float32)

    # obtain age category and noise data for the first batch
    y_batch = cats[0:batch_size]
    z_noise = np.random.normal(0, 1, size = (batch_size, noise))

    # conduct prediction with the generator now
    gen_images = generator.predict_on_batch([z_noise, y_batch])
    
    # save the first 5 image files
    for i, img in enumerate(gen_images[:5]):
        path = os.path.join(OUT_DIR, "img_{}_{}".format(epoch, i))
        save_rgb_img(img, path)


# main GAN train function
if TRAIN_GAN:
    print('\n=== Start GAN training')

    for epoch in range(epochs):
        print("\nEpoch:{}".format(epoch))

        # calculate the total number of iterations needed in each epoch
        iter = int(len(images) / batch_size)

        for index in range(iter):
            # fetch the next batch of images
            img_batch = images[index * batch_size:(index + 1) * batch_size]
            img_batch = img_batch / 127.5 - 1.0
            img_batch = img_batch.astype(np.float32)

            # age category data
            y_batch = cats[index * batch_size:(index + 1) * batch_size]

            # random noise for generator input
            z_noise = np.random.normal(0, 1, size = (batch_size, noise))

            # call discriminator training function
            d_loss = train_disc(img_batch, y_batch, z_noise)

            # call generator training function
            g_loss = train_gen()

            print('   Batch:', index, ', d_loss: %.4f' % d_loss, 
                ', g_loss: %.4f' % g_loss)
        test_gen(epoch)

    # save networks
    generator.save_weights(os.path.join(OUT_DIR, "generator.h5"))
    discriminator.save_weights(os.path.join(OUT_DIR, "discriminator.h5"))


#################
# TRAIN ENCODER #
#################

# train encoder using trained generator
# encoder maps images to noise 
if TRAIN_ENCODER:
    print('\n=== Start encoder training')

    # build and compile encoder
    encoder = build_encoder()
    encoder.compile(loss = euclidean_loss, optimizer = 'adam')

    # load generator weights
    generator.load_weights(os.path.join(OUT_DIR, "generator.h5"))

    # generate random noise input
    noise = np.random.normal(0, 1, size = (num_data, noise))
    print('Noise input shape:', noise.shape)

    # generate random age category input
    age = np.random.randint(0, num_class, size = (num_data,))
    age = np.reshape(np.array(age), [num_data, 1])
    age = to_categorical(age, num_class)
    print('Age input shape:', age.shape)

    # perform epochs and batches
    for epoch in range(epochs):
        print("Epoch:", epoch)

        # compute the number of iterations in this epoch
        iter = int(num_data / batch_size)

        for index in range(iter):

            # fetch the next batch
            z_batch = noise[index * batch_size:(index + 1) * batch_size]
            y_batch = age[index * batch_size:(index + 1) * batch_size]

            # fake images by generator
            fakes = generator.predict_on_batch([z_batch, y_batch])

            # train encoder that maps images into noise
            e_loss = encoder.train_on_batch(fakes, z_batch)

            print('   Batch:', index, ', e_loss: %.4f' % e_loss)

    # save the encoder model
    encoder.save_weights(os.path.join(OUT_DIR, "encoder.h5"))


##################################
# OPTIMIZE ENCODER AND GENERATOR #
##################################

# increase the size of image by 3x
def build_image_resizer():
    input_layer = Input(shape = (64, 64, 3))

    resized_images = Lambda(lambda x: K.resize_images(x, 
        height_factor = 3, width_factor = 3, 
        data_format = 'channels_last'))(input_layer)

    model = Model(input = [input_layer], outputs = [resized_images])

    print('\n=== Image resizer summary')
    model.summary()

    return model


# build face recognition model
# we use pre-trained resnet v2
# input: 192 x 192 x 3 image
# output: 128 values
def build_fr_model(input_shape):

    # borrow resnet
    resnet_model = InceptionResNetV2(include_top = False, weights = 'imagenet', 
        input_shape = input_shape, pooling = 'avg')

    # embedder model
    image_input = resnet_model.input
    x = resnet_model.layers[-1].output
    out = Dense(128)(x)
    embedder_model = Model(inputs = [image_input], outputs = [out])

    # recognizer model
    input_layer = Input(shape = input_shape)
    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis = -1))(x)
    model = Model(inputs = [input_layer], outputs = [output])

    print('\n=== Resnet face recognizer summary')
    model.summary()

    return model


# new GAN with encoder, generator, resizer, and recognizer
# encoder and generator are pre-trained
def build_new_GAN():
    # load the encoder network
    encoder = build_encoder()
    encoder.load_weights(os.path.join(OUT_DIR, "encoder.h5"))

    # load the generator network
    generator.load_weights(os.path.join(OUT_DIR, "generator.h5"))

    # build image resizer network
    image_resizer = build_image_resizer()
    image_resizer.compile(loss = ['binary_crossentropy'], optimizer = 'adam')

    # face recognition model
    fr_model = build_fr_model(input_shape = fr_image_shape)
    fr_model.compile(loss = ['binary_crossentropy'], optimizer = "adam")

    # make the face recognition network as non-trainable
    fr_model.trainable = False

    # input layers
    input_image = Input(shape = image_shape)
    input_label = Input(shape = (num_class,))

    # combine encoder and generator 
    latent = encoder(input_image)
    gen_images = generator([latent, input_label])

    # increase the image size by 3x
    resized_images = Lambda(lambda x: K.resize_images(gen_images, 
        height_factor = 3, width_factor = 3,
        data_format = 'channels_last'))(gen_images)

    # give the enlarged image to resnet to obtain 128 features
    embeddings = fr_model(resized_images)

    # create the final GAN model
    fr_GAN_model = Model(inputs = [input_image, input_label], outputs = [embeddings])
    print('\n=== Face recognizer GAN summary')
    fr_GAN_model.summary()

    # compile the model
    fr_GAN_model.compile(loss = euclidean_loss, optimizer = 'adam')

    return image_resizer, fr_model, fr_GAN_model


# training the new GAN so that the 128 feature extracted by GAN
# becomes similar to the 128 features extracted by resnet
# both encoder and generator improve through this process
if TRAIN_GAN_WITH_FR:

    # build a new GAN using encoder, generator, and resnet
    image_resizer, fr_model, fr_GAN_model = build_new_GAN()

    for epoch in range(epochs):
        print("Epoch:", epoch)

        # calculate the total number of iterations needed in each epoch
        iter = int(len(images) / batch_size)

        for index in range(iter):

            # fetch the next batch of images
            img_batch = images[index * batch_size:(index + 1) * batch_size]
            img_batch = img_batch / 127.5 - 1.0
            img_batch = img_batch.astype(np.float32)

            # age category data
            y_batch = cats[index * batch_size:(index + 1) * batch_size]

            # resized images
            img_batch_resized = image_resizer.predict_on_batch(img_batch)

            # embedding with resnet
            real = fr_model.predict_on_batch(img_batch_resized)

            # train the new GAN and compute loss
            loss = fr_GAN_model.train_on_batch([img_batch, y_batch], real)

            print('   Batch:', index, ', loss: %.4f' % loss)

        # test generator and save images
        test_gen(epoch)
