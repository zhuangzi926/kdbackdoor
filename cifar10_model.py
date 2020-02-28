import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add
from keras.layers import AveragePooling2D, InputLayer, Flatten, MaxPool2D, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model, Sequential
import numpy as np


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


class Teacher_model(tf.keras.Model):
    """teacher net

    Name: ResNet-V1-50
    Link: https://keras.io/examples/cifar10_resnet/
    """

    def __init__(self, depth, num_classes=10):
        super(Teacher_model, self).__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        self.depth = depth
        self.num_classes = num_classes

    def call(self, inputs):
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = InputLayer(input_shape=(32, 32, 3))(inputs)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(x, num_filters=num_filters, strides=strides)
                y = resnet_layer(y, num_filters=num_filters, activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(
                        x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = Add()([x, y])
                x = Activation("relu")(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)

        # Attention! Here returns logits, no activation!
        outputs = Dense(
            self.num_classes, activation=None, kernel_initializer="he_normal"
        )(y)

        return outputs


class Student_model(tf.keras.Model):
    """student net

    Name: a simple deep CNN
    Link: https://keras.io/examples/cifar10_cnn/
    """

    def __init__(self):
        super(Student_model, self).__init__()
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        self.cnn = model

    def call(self, inputs):
        return self.cnn(inputs)


if __name__ == "__main__":
    teacher = Teacher_model(depth=50, num_classes=10)
    student = Student_model()
    img = np.random.random(size=(1, 32, 32, 3))
    print(teacher(img))
    print(teacher.summary())
    print(student(img))
    print(student.summary())
