from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, InputLayer, Flatten, MaxPooling2D, Dropout, Input, Reshape, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import glorot_normal, RandomNormal, Zeros
from tensorflow.keras import backend as K

import settings

def get_model():
    """return 6-layer cnn(tf.keras model)
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=settings.IMG_SHAPE))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))

    return model