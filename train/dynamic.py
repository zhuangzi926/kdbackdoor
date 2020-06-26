import tensorflow as tf
import numpy as np
from tqdm import tqdm

import settings
import train
import datasets


def pretrain(model, dataset_train, dataset_test):
    """pretrain model before distillation"""
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=settings.SGD_LR[0], momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset_train, epochs=settings.NUM_PRETRAIN_EPOCHS)
    loss, acc = model.evaluate(dataset_test)

    return loss, acc


def train_epoch(
    models,
    dataset,
    optimizers,
    batch_size=settings.BATCH_SIZE,
    num_epochs=settings.NUM_EPOCHS,
    temperature=settings.TEMPERATURE,
    loss_teacher_sum=0.0,
    loss_student_sum=0.0,
    loss_backdoor_sum=0.0,
):
    y_target = tf.one_hot(
        models["backdoor"].target_label, depth=settings.NUM_CLASSES, dtype=tf.float32
    )
    y_target = tf.expand_dims(y_target, 0)
    y_target = tf.tile(y_target, tf.constant([batch_size, 1], tf.int32))

    for (batch_index, (x, y)) in tqdm(
        dataset.enumerate(),
        desc="epoch",
        unit=" batch",
        leave=False,
        total=settings.NUM_TRAIN_DATA // settings.BATCH_SIZE,
    ):
        backdoored_x = models["backdoor"](x)

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

        backdoored_x = tf.map_fn(
            lambda data: datasets.utils.augment(data[0], data[1])[0],
            (backdoored_x, y),
            dtype=tf.float32,
        )
        backdoored_x = tf.map_fn(
            lambda data: datasets.utils.normalize(data[0], data[1])[0],
            (backdoored_x, y),
            dtype=tf.float32,
        )

        # Train teacher
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(models["teacher"].trainable_weights)
            logits_from_benign = models["teacher"](x)
            logits_from_backdoor = models["teacher"](backdoored_x)
            loss_teacher = train.losses.loss_teacher.loss_fn(
                logits_from_benign,
                logits_from_backdoor,
                y,
                y_target,
                tf.constant(temperature, tf.float32),
            )
        grads = tape.gradient(loss_teacher, models["teacher"].trainable_weights)
        optimizers["teacher"].apply_gradients(
            grads_and_vars=zip(grads, models["teacher"].trainable_weights)
        )
        loss_teacher_sum += loss_teacher

        # Train student
        # if batch_index % 2 == 1:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(models["student"].trainable_weights)
            logits_from_teacher = models["teacher"](x)
            logits_from_student = models["student"](x)
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

        # Train backdoor
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(models["backdoor"].trainable_weights)
            loss_backdoor = train.losses.loss_backdoor.loss_fn(
                models, backdoored_x, y_target
            )
        grads = tape.gradient(loss_backdoor, models["backdoor"].trainable_weights)
        optimizers["backdoor"].apply_gradients(
            grads_and_vars=zip(grads, models["backdoor"].trainable_weights)
        )
        loss_backdoor_sum += loss_backdoor
    return (
        loss_teacher_sum / num_epochs,
        loss_student_sum / num_epochs,
        loss_backdoor_sum / num_epochs,
    )

