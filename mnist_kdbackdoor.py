import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import os
import datetime

from mnist_loader import MNISTLoader
from mnist_model import Teacher_model, Student_model

# log dir
log_dir = os.path.join('./logs', datetime.datetime.now().strftime("%Y%m%d%H%M"))
if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

# model dir
model_dir = os.path.join('./models', datetime.datetime.now().strftime("%Y%m%d%H%M"))
if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)


@tf.function
def softmax_with_temperature(logits, temperature):
    """softmax function with temperature

    Args:
        logits: output of neurons from last layer, shape = (batch_size, class_num)
        temperature: hyper parameters to control knowledge distillation
    
    Return:
        x: result of activation, shape = (batch_size, class_num)
    """

    x = logits / temperature
    expx = tf.exp(x)
    sum_exp = tf.reduce_mean(expx, axis=1)
    sum_exp = tf.expand_dims(sum_exp, axis=-1)
    x = expx / sum_exp
    return x


@tf.function
def loss_student_fn(y_pred_student, y_pred_teacher, y, temperature):
    loss_student = keras.losses.categorical_crossentropy(
        y_true=softmax_with_temperature(y_pred_teacher, temperature),
        y_pred=softmax_with_temperature(y_pred_student, temperature),
        from_logits=False,
        label_smoothing=0,
    )
    loss_student *= temperature ** 2
    loss_student += keras.losses.categorical_crossentropy(
        y_true=y, y_pred=y_pred_student, from_logits=True, label_smoothing=0
    )
    loss_student = tf.reduce_mean(loss_student)
    return loss_student


# hyper parameters
num_epochs = 3
batch_size = 32
learning_rate_teacher = 1e-3
learning_rate_student = 2e-3
temperature = 20

# load dataset
data_loader = MNISTLoader()
data_loader.preprocess()
data_generator = data_loader.get_batch(batch_size, training=True)

# build model
teacher = Teacher_model()
student = Student_model()

# optimizer
optimizer_teacher = tf.keras.optimizers.Adam(learning_rate=learning_rate_teacher)
optimizer_student = tf.keras.optimizers.Adam(learning_rate=learning_rate_student)

# customize training
for epoch_index in range(num_epochs):

    print("epoch: ", epoch_index + 1)

    num_batches = data_loader.num_train_data // batch_size
    for batch_index in range(num_batches):
        X, y = next(data_generator)

        # train teacher
        with tf.GradientTape() as tape:
            y_pred_teacher = teacher(X)
            y_pred_teacher = tf.nn.softmax(y_pred_teacher)
            loss_teacher = tf.reduce_mean(
                keras.losses.categorical_crossentropy(
                    y_true=y, y_pred=y_pred_teacher, from_logits=False, label_smoothing=0
                )
            )
        grads = tape.gradient(loss_teacher, teacher.trainable_weights)
        optimizer_teacher.apply_gradients(
            grads_and_vars=zip(grads, teacher.trainable_weights)
        )

        # train student
        if batch_index % 2:
            with tf.GradientTape() as tape:
                y_pred_student = student(X)
                y_pred_teacher = teacher(X)
                loss_student = loss_student_fn(
                    y_pred_student, y_pred_teacher, y, temperature
                )
            grads = tape.gradient(loss_student, student.trainable_weights)
            optimizer_student.apply_gradients(
                grads_and_vars=zip(grads, student.trainable_weights)
            )

# evaluate model
accuracy_teacher = tf.keras.metrics.CategoricalAccuracy()
accuracy_student = tf.keras.metrics.CategoricalAccuracy()
data_generator = data_loader.get_batch(batch_size, training=False)
num_batches = data_loader.num_test_data // batch_size
for batch_index in range(num_batches):
    X, y = next(data_generator)

    y_pred_teacher = teacher(X)
    accuracy_teacher.update_state(y, y_pred_teacher)

    y_pred_student = student(X)
    accuracy_student.update_state(y, y_pred_student)
print("Teacher Acc: ", accuracy_teacher.result().numpy())
print("Student Acc: ", accuracy_student.result().numpy())

