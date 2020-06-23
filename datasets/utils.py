import tensorflow as tf
import numpy as np

import settings


def convert(image, label):
    """from nint8 to float32, from [0, 255] to [0, 1]"""
    image = tf.image.convert_image_dtype(image, tf.float32)  # to [0, 1]
    label = tf.squeeze(tf.one_hot(label, depth=settings.NUM_CLASSES, dtype=tf.float32))

    return image, label


def augment(image, label):
    """data augment"""
    image = tf.image.resize_with_crop_or_pad(image, settings.IMG_HEIGHT + 4, settings.IMG_WIDTH + 4)
    image = tf.image.random_crop(image, size=settings.IMG_SHAPE)
    image = tf.image.random_flip_left_right(image)

    return image, label


def normalize(image, label):
    """ img = (img - mean) / std """
    image -= settings.DATASET_MEAN
    image /= settings.DATASET_STD

    return image, label


def preprocess_train(image, label):
    # Rescale
    image, label = convert(image, label)

    # Augment
    image, label = augment(image, label)

    # Normalize
    image, label = normalize(image, label)

    return image, label


def preprocess_test(image, label):
    # Rescale
    image, label = convert(image, label)

    # Normalize
    image, label = normalize(image, label)

    return image, label
