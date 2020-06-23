import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class MNISTLoader:
    def __init__(self):
        mnist = keras.datasets.mnist
        # MNIST中的图像默认为uint8（0-255的数字）
        # 在最后增加一维作为颜色通道
        (
            (self.train_data, self.train_label),
            (self.test_data, self.test_label),
        ) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data, axis=-1)
        self.test_data = np.expand_dims(self.test_data, axis=-1)
        self.num_train_data, self.num_test_data = (
            self.train_data.shape[0],
            self.test_data.shape[0],
        )

    def preprocess(self):
        """数据预处理，归一化到[0, 1]
		"""

        class_num = np.max(self.train_label) + 1

        self.train_data = self.train_data.astype(np.float32) / 255.0
        self.test_data = self.test_data.astype(np.float32) / 255.0

        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]

    def get_batch(self, batch_size=32, training=True):
        """fetch train data and one-hot label from dataset

        Return: a generator of train/test dataset
        """
        if training:
            data = self.train_data
            label = self.train_label
            num = self.num_train_data
        else:
            data = self.test_data
            label = self.test_label
            num = self.num_test_data
        random_index = np.arange(num)
        np.random.shuffle(random_index)
        i = 0
        while True:
            if i + batch_size >= num:
                i = 0
                np.random.shuffle(random_index)
                continue
            index = random_index[i : i + batch_size]
            X = data[index, :]
            y = tf.one_hot(label[index], depth=10)
            yield X, y
            i += batch_size
