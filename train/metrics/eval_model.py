import tensorflow as tf
import numpy as np


def acc(model, dataset):
    """evaluate accuracy of model

    Args:
        model(tf.keras.Model)
        dataset(tf.data.Dataset): test dataset

    Returns:
        accuracy(tensor)
    """
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    for (batch_index, (x, y)) in dataset.enumerate():
        y_pred = tf.nn.softmax(model(x))
        accuracy.update_state(y, y_pred)

    return accuracy.result()


def loss(model, dataset):
    """evaluate cross-entropy loss of model

    Args:
        model(tf.keras.Model)
        dataset(tf.data.Dataset): test dataset

    Returns:
        loss(tensor)
    """
    loss = tf.keras.metrics.CategoricalCrossentropy()
    for (batch_index, (x, y)) in dataset.enumerate():
        y_pred = tf.nn.softmax(model(x))
        loss.update_state(y, y_pred)

    return loss.result()
