import os
import scipy
import numpy as np
import tensorflow as tf


#### ASDF ####
import matplotlib.pyplot as plt
import sys
import imutils
import cv2
from config import cfg
from scipy.io.matlab import loadmat
import sklearn.preprocessing
import time

def load_affnist(path):
    """
        Index 2 contains the images.
        An image is given by data[2][:,sample]
        because data[2] has shape (1600, 320000)
        A Label is given by data[5][:,sample]
        A one hot label is given by data[4][:,sample]
    """

    # Create a dictionary containing the data and extract the data.
    mat_dict = loadmat(path)
    data = mat_dict['affNISTdata']

    # Remove redundant arrays surrounding the data.
    data = data[0][0]
    assert len(data) == 8

    # The number of samples in this set.
    num_samples = len(data[0][0])

    # Loop over all the samples.
    train = []
    label = []
    for i in range(num_samples):
        train.append(np.array(data[2][:,i]))
        label.append(data[5][:,i])

    # Reshape into the expect format.
    train = np.reshape(train, (num_samples, 40, 40, 1))
    label = np.reshape(label, (num_samples))

    # Normalize.
    train = train / 255.

    return train, label

def random_translation(image, horizontal_max=6, vertical_max=6):

    # Remove the channel.
    image = np.squeeze(image)

    # Randomly apply translation to image.
    horizontal_translation = np.random.randint(-horizontal_max, horizontal_max+1)
    vertical_translation = np.random.randint(-vertical_max, vertical_max+1)
    translated = imutils.translate(image, horizontal_translation, vertical_translation)

    # Add the channel.
    image = np.reshape(translated, (image.shape[0], image.shape[1], 1))
    
    return image

def rotate_grayscale_image(image):

    # Remove the channel.
    image = np.squeeze(image)

    # Rotate by an angle.
    rotation_angle = np.random.randint(-cfg.max_rotation_angle, cfg.max_rotation_angle)
    # rotation_angle = np.random.choice((-cfg.max_rotation_angle, cfg.max_rotation_angle))
    rotated = imutils.rotate(image, rotation_angle)

    # Add the channel.
    image = np.reshape(rotated, (image.shape[0], image.shape[1], 1))
    
    return image

def pad_grayscale_image_with_zero(image):

    # Remove the channel.
    image = np.squeeze(image)

    # Pad with zero.
    pixels_to_pad = (cfg.image_size - 28)//2
    padded = cv2.copyMakeBorder(image, pixels_to_pad, pixels_to_pad, pixels_to_pad, pixels_to_pad, cv2.BORDER_CONSTANT, value=0.0)

    # Add the channel.
    image = np.reshape(padded, (cfg.image_size, cfg.image_size, 1))

    return image

def create_multimnist(is_training):

    # Set the random number seed.
    np.random.seed(0)

    # Base path.
    path = os.path.join('data', 'mnist')

    # Load the data.
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        dataX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        dataY = loaded[8:].reshape((60000)).astype(np.int32)
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        dataX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
     
        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        dataY = loaded[8:].reshape((10000)).astype(np.int32)

    # Obtain the size of the dataset.
    size_dataset = len(dataX)

    # Create label binarizer to one hot encode the digits.
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(10))

    # Container to hold results.
    tmp_arrayX = []
    tmp_arrayY = []

    # Define the number of images that we wish to sample.
    num_sampled_images_needed = cfg.num_sampled_images_needed_multimnist

    # Loop over the images.
    for index, image in enumerate(dataX):

        # Obtain the label of the image.
        image_label = dataY[index]

        # Pad and translate the mnist image to 36x36.
        # Be sure to set in the configuration file to have a size of 36.
        image = pad_grayscale_image_with_zero(image)

        # Translate image by some pixels.
        image = random_translation(image, cfg.max_translation_pixel, cfg.max_translation_pixel)

        # Binarize the image label.
        image_one_hot_label = np.squeeze(label_binarizer.transform([dataY[index]]))

        # Instantiate the number of images we have sampled for this base image.
        num_sampled_images = 0

        # Instantiate the set of already sampled images.
        sampled_set = set()

        # Keep sampling images until we have enough images.
        while(num_sampled_images < num_sampled_images_needed):

            # Sample an image from the images.
            sample_index = np.random.randint(size_dataset)

            # Obtain its label.
            sample_label = dataY[sample_index]

            # If the label of the image is the same as the base image, or if we have
            # already sampled that particular image, then break the loop.
            if image_label == sample_label or sample_index in sampled_set:
                continue
            else:
                # Increment the counter.
                num_sampled_images+=1

                # Add the image to the sampled set.
                sampled_set.add(sample_index)

            # Obtain the sampled image.
            sampled_image = dataX[sample_index]

            # Pad and translate the mnist image to 36x36.
            # Be sure to set in the configuration file to have a size of 36.
            sampled_image = pad_grayscale_image_with_zero(sampled_image)

            # Translate the image by some pixels.
            sampled_image = random_translation(sampled_image, cfg.max_translation_pixel, cfg.max_translation_pixel)

            # Add the two images.
            tmp_arrayX.append(sampled_image + image)

            # Binarizer the labels of the sample.
            one_hot_sample_label = np.squeeze(label_binarizer.transform([sample_label]))

            # Concatenate the two labels and append to an array.
            # i.e. a "2" and "3" would look like [0.0 0.0 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.0]
            tmp_arrayY.append((image_one_hot_label + one_hot_sample_label)/2.0)

    # Set the tmp arrays back to the official arrays.
    dataX = np.array(tmp_arrayX)
    dataY = np.array(tmp_arrayY)

    # Normalize.
    dataX = dataX / 510.

    # Save the data.
    if is_training:
        np.save(cfg.multimnist_path + 'multimnist_trainX' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy', dataX)
        np.save(cfg.multimnist_path + 'multimnist_trainY' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy', dataY)
    else:
        np.save(cfg.multimnist_path + 'multimnist_testX' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy', dataX)
        np.save(cfg.multimnist_path + 'multimnist_testY' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy', dataY)   

def load_multimnist(is_training):
    if is_training:

        # Load from disk.
        dataX = np.load(cfg.multimnist_path + 'multimnist_trainX' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy')
        dataY = np.load(cfg.multimnist_path + 'multimnist_trainY' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy')

        assert len(dataX) == len(dataY)

        # Seperate into training and validation sets.
        num_training_samples = int(len(dataX)*55./60.)
        trX = dataX[:num_training_samples]
        trY = dataY[:num_training_samples]
        valX = dataX[num_training_samples:]
        valY = dataY[num_training_samples:]

        # Determine the number of batches given the batch size and the data sizes.
        num_tr_batch = len(trX) // cfg.batch_size
        num_val_batch = len(valX) // cfg.batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch

    else:

        # Load from disk.
        testX = np.load(cfg.multimnist_path + 'multimnist_testX' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy')
        testY = np.load(cfg.multimnist_path + 'multimnist_testY' + '_' + str(cfg.num_sampled_images_needed_multimnist) + '.npy')

        assert len(testX) == len(testY)

        # Determine the number of batches given the batch size and the data sizes.
        num_test_batch = len(testX) // cfg.batch_size

        return testX, testY, num_test_batch

def load_multimnist_small(is_training):
    if is_training:

        # Load from disc.
        dataX = np.load(cfg.multimnist_path + 'multimnist_trainX_small.npy')
        dataY = np.load(cfg.multimnist_path + 'multimnist_trainY_small.npy')

        assert len(dataX) == len(dataY)

        # Seperate into training and validation sets.
        num_training_samples = int(len(dataX)*55./60.)
        trX = dataX[:num_training_samples]
        trY = dataY[:num_training_samples]
        valX = dataX[num_training_samples:]
        valY = dataY[num_training_samples:]

        # Determine the number of batches given the batch size and the data sizes.
        num_tr_batch = len(trX) // cfg.batch_size
        num_val_batch = len(valX) // cfg.batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch

    else:
        testX = np.load(cfg.multimnist_path + 'multimnist_testX_small.npy')
        testY = np.load(cfg.multimnist_path + 'multimnist_testY_small.npy')

        assert len(testX) == len(testY)

        # Determine the number of batches given the batch size and the data sizes.
        num_test_batch = len(testX) // cfg.batch_size

        return testX, testY, num_test_batch

def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')

    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)
    
        # Loop over images for various image manipulation operations.
        np.random.seed(0)
        tmp_array = []
        for image in trainX:

            # Pad training images with zeros.
            if cfg.use_image_transformation_train_pad_zero:                
                image = pad_grayscale_image_with_zero(image)

            # Apply random rotation.
            if cfg.use_image_transformation_train_rotation:                
                image = rotate_grayscale_image(image)

            # Apply random translation.
            if cfg.use_image_transformation_train_translation:
                image = random_translation(image)

            # Add the image to the temp array.
            tmp_array.append(image)

        # Convert to a numpy array.    
        trainX = np.array(tmp_array)

        # Normalize and divide into test and validation sets.
        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:] / 255.
        valY = trainY[55000:]

        # Reduce the dataset into a 10 image data set.
        if cfg.use_10_images:
            unique_images = 5

        # Reduce the training data size.
        amount_train_use = int(len(trX)*cfg.percent_train_use)
        trX = trX[:amount_train_use]
        trY = trY[:amount_train_use]
        
        # Compute the number of training and validation batches.
        num_tr_batch = len(trX) // batch_size
        num_val_batch = len(valX) // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        teX = teX / 255.

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        # Loop over images for various image manipulation operations.
        np.random.seed(0)
        tmp_array = []
        for image in teX:

            # Pad training images with zeros.
            if cfg.use_image_transformation_test_pad_zero:
                image = pad_grayscale_image_with_zero(image)

            # Apply random rotation.
            if cfg.use_image_transformation_test_rotation:
                image = rotate_grayscale_image(image)

            # Apply random translation.
            if cfg.use_image_transformation_test_translation:            
                image = random_translation(image)

            # Add the image to the temp array.
            tmp_array.append(image)

        # Convert to a numpy array.    
        teX = np.array(tmp_array)


        # Replace contents of data with affnist set.
        if cfg.use_affnist_test:
            teX, teY = load_affnist(cfg.affNistTest)

        num_te_batch = len(teX) // batch_size
        return teX, teY, num_te_batch

def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)

def get_batch_data_multimnist(batch_size, num_threads):
    
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_multimnist_small(is_training=True)
    
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)

def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
