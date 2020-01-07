# MNIST Hand-written Digit Generation with Semi-Supervied GAN
# Our GAN is built using CNNs
# January 4, 2020
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
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.layers import Activation, UpSampling2D, Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing import image
from scipy.io import loadmat
from keras.models import Sequential


# directories
DB_DIR = "./dataset/wiki_crop/"
OUT_DIR = "./ch3-output/"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# hyper-parameters and constants
# for good results: epoch = 500, batch = 100
epochs = 10
batch_size = 100
noise = 100
num_class = 6
num_data = 1000
DB_NAME = 'ch3-1000.mat'
image_shape = (64, 64, 3)
fr_image_shape = (192, 192, 3)


# program control option 1: on-off-off
# program control option 2: on-on-off then off-off-on
TRAIN_GAN = False
TRAIN_ENCODER = False
TRAIN_GAN_WITH_FR = True


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

    # flattend to get 8192 neurons
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
# we downselect 1000 images out of 62328 
def load_data():

    # Load the wiki.mat file
    # meta contains 4 dimesional dictionary
    dataset = 'wiki'
    meta = loadmat(os.path.join(DB_DIR, DB_NAME))
    full_path = meta[dataset][0, 0]["full_path"][0]
    print('\nWe use', len(full_path), 'images.')


    # List of date-of-birth and year-taken data
    dob = meta[dataset][0, 0]["dob"][0]
    photo_taken = meta[dataset][0, 0]["photo_taken"][0]


    # Calculate age for all data
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]


    # Create a list of tuples containing a pair of an image path and age
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
        try:
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

        except Exception as e:
            print("Error:", i, e)
    print("Image loading done.")

    return images


# Euclidean distance loss
# https://en.wikipedia.org/wiki/Euclidean_distance
# :param y_true: TensorFlow/Theano tensor
# :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
# :return: float
def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# save the given color iamge to the output directory
def save_rgb_img(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
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
    # Generate fake images
    fakes = generator.predict_on_batch([z_noise, y_batch])

    # implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype = np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype = np.float32) * 0.1
    
    # train discrimonator using real and fake images
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


# test the generator with the first batch
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
        save_rgb_img(img, path = "ch3-output/img_{}_{}.png".format(epoch, i))
        

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

            print('   Iter:', index, ', d_loss: %.4f' % d_loss, 
                ', g_loss: %.4f' % g_loss)
        test_gen(epoch)

    # save networks
    generator.save_weights("ch3-output/generator.h5")
    discriminator.save_weights("ch3-output/discriminator.h5")


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
    generator.load_weights("ch3-output/generator.h5")

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

        iter = int(num_data / batch_size)
        for index in range(iter):

            # fetch the next batch
            z_batch = noise[index * batch_size:(index + 1) * batch_size]
            y_batch = age[index * batch_size:(index + 1) * batch_size]

            # fake images by generator
            fakes = generator.predict_on_batch([z_batch, y_batch])

            # train encoder that maps images into noise
            e_loss = encoder.train_on_batch(fakes, z_batch)

            print('   Iter:', index, ', e_loss: %.4f' % e_loss)

    # save the encoder model
    encoder.save_weights("ch3-output/encoder.h5")


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

    print('\n=== IMAGE RESIZER summary')
    model.summary()

    return model


# build face recognition model
# we use pre-trained resnet v2
# input: 192 x 192 x 3 image
# output: 128 values
def build_fr_model(input_shape):
    resnet_model = InceptionResNetV2(include_top = False, weights = 'imagenet', 
        input_shape = input_shape, pooling = 'avg')
    image_input = resnet_model.input
    x = resnet_model.layers[-1].output
    out = Dense(128)(x)

    embedder_model = Model(inputs = [image_input], outputs = [out])

    input_layer = Input(shape = input_shape)

    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis = -1))(x)

    model = Model(inputs=[input_layer], outputs=[output])
    print('\n=== Face recognizer summary')
    model.summary()
    exit()

    return model


if TRAIN_GAN_WITH_FR:

    # load the encoder network
    encoder = build_encoder()
    encoder.load_weights("ch3-output/encoder.h5")

    # load the generator network
    generator.load_weights("ch3-output/generator.h5")

    image_resizer = build_image_resizer()
    image_resizer.compile(loss = ['binary_crossentropy'], optimizer = 'adam')

    # face recognition model
    fr_model = build_fr_model(input_shape = fr_image_shape)
    fr_model.compile(loss = ['binary_crossentropy'], optimizer = "adam")

    # make the face recognition network as non-trainable
    fr_model.trainable = False

    # input layers
    input_image = Input(shape=(64, 64, 3))
    input_label = Input(shape=(6,))

    # Use the encoder and the generator network
    latent0 = encoder(input_image)
    gen_images = generator([latent0, input_label])

    # Resize images to the desired shape
    resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=3, width_factor=3,
                                                        data_format='channels_last'))(gen_images)
    embeddings = fr_model(resized_images)

    # Create a Keras model and specify the inputs and outputs for the network
    fr_GAN_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

    # Compile the model
    fr_GAN_model.compile(loss=euclidean_loss, optimizer='adam')

    for epoch in range(epochs):
        print("Epoch:", epoch)

        reconstruction_losses = []

        iter = int(len(images) / batch_size)
        print("Number of batches:", iter)
        for index in range(iter):
            print("Batch:", index + 1)

            img_batch = images[index * batch_size:(index + 1) * batch_size]
            img_batch = img_batch / 127.5 - 1.0
            img_batch = img_batch.astype(np.float32)

            y_batch = cats[index * batch_size:(index + 1) * batch_size]

            img_batch_resized = image_resizer.predict_on_batch(img_batch)

            real_embeddings = fr_model.predict_on_batch(img_batch_resized)

            reconstruction_loss = fr_GAN_model.train_on_batch([img_batch, y_batch], real_embeddings)

            print("Reconstruction loss:", reconstruction_loss)

            reconstruction_losses.append(reconstruction_loss)

        img_batch = images[0:batch_size]
        img_batch = img_batch / 127.5 - 1.0
        img_batch = img_batch.astype(np.float32)

        y_batch = cats[0:batch_size]
        z_noise = np.random.normal(0, 1, size=(batch_size, noise))

        gen_images = generator.predict_on_batch([z_noise, y_batch])

        for i, img in enumerate(gen_images[:5]):
            save_rgb_img(img, path="ch3-output/img_opt_{}_{}.png".format(epoch, i))

    # Save improved weights for both of the networks
    generator.save_weights("ch3-output/generator_optimized.h5")
    encoder.save_weights("ch3-output/encoder_optimized.h5")
