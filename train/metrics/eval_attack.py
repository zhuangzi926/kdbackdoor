import tensorflow as tf
import numpy as np

import settings
import datasets



def succ(model, backdoor, dataset):
    """evaluate attack success rate

    Args:
        model(tf.keras.Model)
        backdoor(tf.keras.Model)
        dataset(tf.data.Dataset): test dataset

    Returns:
        attack success rate(tensor)
    """
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    y_target = tf.one_hot(
        backdoor.target_label, depth=settings.NUM_CLASSES, dtype=tf.float32
    )
    y_target = tf.expand_dims(y_target, 0)
    y_target = tf.tile(y_target, tf.constant([settings.BATCH_SIZE, 1], tf.int32))
    for (batch_index, (x, y)) in dataset.enumerate():
        backdoored_x = backdoor(x)
        backdoored_x = tf.map_fn(
            lambda data: datasets.utils.normalize(data[0], data[1])[0],
            (backdoored_x, y),
            dtype=tf.float32,
        )


        y_pred = model(backdoored_x)
        accuracy.update_state(y_target, y_pred)

    return accuracy.result()
