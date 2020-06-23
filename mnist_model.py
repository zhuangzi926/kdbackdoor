import tensorflow as tf
import tensorflow.keras as keras


class Teacher_model(tf.keras.Model):
    """teacher net
    """

    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        )
        self.layer2 = keras.layers.Conv2D(
            filters=128,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        )
        self.layer3 = keras.layers.Flatten()
        self.layer4 = keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.layer5 = keras.layers.Dense(units=10, activation=None)

    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.layer5(x)
        return output


class Student_model(tf.keras.Model):
    """student net
    """

    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Flatten()
        self.layer2 = keras.layers.Dense(units=800, activation=tf.nn.leaky_relu)
        self.layer3 = keras.layers.Dense(units=800, activation=tf.nn.leaky_relu)
        self.layer4 = keras.layers.Dense(units=10, activation=None)

    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)
        return output

