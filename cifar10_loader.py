import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class CIFAR10_loader:
    """cifar10 train/test dataset

    Attributes:
        dataset_train(tf dataset): a dataset for auto iter, return (x, y), x in [-1, 1]
        dataset_test(tf dataset): a dataset for auto iter, return (x, y), x in [-1, 1]
        num_train_data(int): number of train records
        num_test_data(int): number of test records
    """

    def __init__(self, batch_size=64):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # normalize
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype=np.float32)
        y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype=np.float32)

        # subtract pixel mean
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        # turn keras dataset into tf dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.shuffle(10000)
        dataset_train = dataset_train.batch(batch_size)

        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset_test = dataset_test.repeat()
        dataset_test = dataset_test.shuffle(10000)
        dataset_test = dataset_test.batch(batch_size)

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        self.num_train_data = 60000
        self.num_test_data = 10000

    def get_gen(self, training=True):
        """return train/test dataset generator

        Args:
            training(bool): True for train data, False for test data
        
        Returns:
            dataset_gen(tf dataset generator): generate tf tensor data from tf dataset
        """

        if training:
            dataset_gen = self.dataset_train.enumerate()
        else:
            dataset_gen = self.dataset_test.enumerate()
        return iter(dataset_gen)


if __name__ == "__main__":
    cifar10_loader = CIFAR10_loader(batch_size=64)
    train = cifar10_loader.get_gen(training=True)
    element = next(train)
    print("train type: ", type(train))
    print(element)
    test = cifar10_loader.get_gen(training=False)
    element = next(test)
    print("test type: ", type(test))
    print(element)
