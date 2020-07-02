import tensorflow as tf
import numpy as np

import settings
import train
import datasets


@tf.function
def train_epoch(
    models,
    dataset,
    optimizers,
    loss_student_sum=0.0,
    temperature=settings.TEMPERATURE,
    num_train_data=settings.NUM_TRAIN_DATA,
    batch_size=settings.BATCH_SIZE,
):
    num_batches = num_train_data // batch_size

    for (batch_index, (x, y)) in dataset.enumerate():
        x = tf.map_fn(
            lambda data: datasets.utils.augment(data[0], data[1])[0],
            (x, y),
            dtype=tf.float32,
        )
        x = tf.map_fn(
            lambda data: datasets.utils.normalize(data[0], data[1])[0],
            (x, y),
            dtype=tf.float32,
        )

        # Train student
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(models["student"].trainable_weights)
            logits_from_student = models["student"](x)
            logits_from_teacher = models["teacher"](x)
            loss_student = train.losses.loss_student.loss_fn(
                logits_from_student,
                logits_from_teacher,
                y,
                tf.constant(temperature, tf.float32),
            )
        grads = tape.gradient(loss_student, models["student"].trainable_weights)
        optimizers["student"].apply_gradients(
            grads_and_vars=zip(grads, models["student"].trainable_weights)
        )
        loss_student_sum += loss_student
    return loss_student_sum / num_batches
