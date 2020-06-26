from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import (
    AveragePooling2D,
    InputLayer,
    Flatten,
    MaxPooling2D,
    Dropout,
    Input,
    Reshape,
    DepthwiseConv2D,
)
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import glorot_normal, RandomNormal, Zeros
from tensorflow.keras import backend as K

import settings


class Injector(tf.keras.layers.Layer):
    """inject backdoor into benign images

	Attributes:
		mask: a trainable tf Variable
		trigger: a non-trainable tf Variable
	"""

    def __init__(self):
        super(Injector, self).__init__()

    def build(self, input_shape):
        self.mask = self.add_weight(
            shape=input_shape[1:].as_list(),
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
            dtype=tf.float32,
            trainable=True,
            name="mask",
        )
        self.trigger = self.add_weight(
            shape=input_shape[1:].as_list(),
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
            dtype=tf.float32,
            trainable=False,
            name="trigger",
        )

    def call(self, inputs):
        """inject backdoor into an image

		backdoored_image = (1 - mask) * image + mask * trigger

		Args:
			inputs: a tensor of images, size = (batch_size, W, H, C), value in [0, 1]
		
		Returns:
			output: a tensor of backdoored images, size = (batch_size, W, H, C), value in [0, 1]
		"""

        outputs = (1 - self.mask) * inputs + self.mask * self.trigger
        outputs = tf.clip_by_value(outputs, 0, 1.0)
        return outputs


class Backdoor(tf.keras.Model):
    """backdoor(poison) attack against neural network
	"""

    def __init__(self, target_label=settings.TARGET_LABEL):
        """generate random trigger
		
		Args:
			target_label: an int indicates the victim label
		"""
        super(Backdoor, self).__init__()
        self.injector = Injector()
        self.target_label = target_label

    def call(self, inputs):
        outputs = self.injector(inputs)
        return outputs

    def get_mask(self):
        return self.injector.mask

    def get_trigger(self):
        return self.injector.trigger


def get_model():
    return Backdoor()

