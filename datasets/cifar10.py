import tensorflow as tf
import numpy as np

import settings
import datasets.utils


class CIFAR10Loader:
    """cifar10 train/test dataset

    Attributes:
        dataset_train(tf.data.Dataset): a dataset for auto iter, return (x, y), x in [0, 1]
        dataset_test(tf.data.Dataset): a dataset for auto iter, return (x, y), x in [0, 1]
        num_train_data(int): number of train data
        num_test_data(int): number of test data
    """

    def __init__(
        self,
        batch_size=settings.BATCH_SIZE,
        num_train_data=settings.NUM_TRAIN_DATA,
        num_test_data=settings.NUM_TEST_DATA,
    ):
        self.batch_size = batch_size
        self.num_train_data = num_train_data
        self.num_test_data = num_test_data

        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

        train_x = np.uint8(train_x)
        test_x = np.uint8(test_x)

        self.dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.dataset_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    def preprocess(self):
        self.dataset_train = (
            self.dataset_train.cache()
            .shuffle(self.num_train_data)
            .map(convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.dataset_test = (
            self.dataset_test.cache()
            .shuffle(self.num_test_data)
            .map(convert, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def get_dataset(self, training=True):
        """return train/test dataset(tf.data.Dataset)
        """
        if training:
            return self.dataset_train
        else:
            return self.dataset_test
