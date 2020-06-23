import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm

import os
import datetime
import logging

from mnist_loader import MNISTLoader
from mnist_model import Teacher_model, Student_model
from mnist_backdoor import Backdoor

# log dir
log_dir = os.path.join('.\logs', datetime.datetime.now().strftime("%Y%m%d%H%M"))
if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

# model dir
model_dir = os.path.join('.\models', datetime.datetime.now().strftime("%Y%m%d%H%M"))
if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

# logging setting
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


@tf.function
def loss_teacher_fn(logits_from_benign, logits_from_backdoor,
                    benign_label, target_label, temperature):
    """loss function of teacher model

    loss_teacher = softmax_with_logits(teacher(X) / T, y) + softmax(teacher(X_t) / T, target)

    Args:
        logits_from_benign: a tensor from output of teacher model, size = (batch_size, class_num)
        logits_from_backdoor: a tensor from backdoored output of teacher model, size = (batch_size, class_num)
        benign_label: a numpy array from dataset, one-hot encoded, size = (batch_size, class_num)
        target_label: a numpy array of target label, one-hot encoded, size = (batch_size, class_num)
        temperature: an int, hyperparameter that controls knowledge distillation
    
    Returns:
        loss_teacher: a float value indicates loss of teacher model
    """

    loss_teacher = tf.nn.softmax_cross_entropy_with_logits(
        labels=benign_label,
        logits=logits_from_benign / temperature
    )
    loss_teacher += tf.nn.softmax_cross_entropy_with_logits(
        labels=target_label,
        logits=logits_from_backdoor / temperature
    )
    return tf.reduce_mean(loss_teacher)


@tf.function
def loss_student_fn(logits_from_student, logits_from_teacher, 
                    benign_label, temperature):
    """loss function of student model

    loss_student = softmax_with_logits(student(X), y) * 0.2
        + softmax_with_logits((student(X) / T), softmax(teacher(X) / T)) * 0.8
    
    Args:
        logits_from_student: a tensor from output of student model, size = (batch_size, class_num)
        logits_from_teacher: a tensor from output of teacher model, size = (batch_size, class_num)
        benign_label: a tensor from dataset, one-hot encoded, size = (batch_size, class_num)
        temperature: an int, hyperparameter that controls knowledge distillation
    
    Returns:
        loss_student: a float value indicates loss of student model
    """
    soft_label_from_teacher = tf.nn.softmax(logits_from_teacher / temperature)
    loss_student = tf.nn.softmax_cross_entropy_with_logits(
        labels=soft_label_from_teacher,
        logits=logits_from_student / temperature
    ) * 0.8
    loss_student += tf.nn.softmax_cross_entropy_with_logits(
        labels=benign_label,
        logits=logits_from_student
    ) * 0.2
    return tf.reduce_mean(loss_student)


@tf.function
def loss_backdoor_fn(model_teacher, model_student, model_backdoor, 
                     X, target_label):
    """loss function of backdoor model

    loss_student = softmax_with_logits(teacher(backdoor(X)), target)
        + softmax_with_logits(student(backdoor(X)), target)
        + L2_norm(mask_matrix)
    
    Args:
        model_teacher: a keras model of teacher
        model_student: a keras model of student
        model_backdoor: a keras model of backdoor
        X: a numpy array of data, size = (batch_size, H, W, C)
        target_label: a numpy array of target label, one-hot encoded, size = (batch_size, class_num)
    
    Returns:
        loss_backdoor: a float value indicates loss of backdoor model
    """

    backdoored_X = model_backdoor(X)
    logits_from_teacher = model_teacher(backdoored_X)
    logits_from_student = model_student(backdoored_X)
    loss_backdoor = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_label,
        logits=logits_from_teacher
    )
    loss_backdoor += tf.nn.softmax_cross_entropy_with_logits(
        labels=target_label,
        logits=logits_from_student
    )
    loss_backdoor += tf.nn.l2_loss(model_backdoor.get_mask())
    return tf.reduce_mean(loss_backdoor)


def evaluate_model_acc(model_teacher, model_student, data_loader, batch_size=128):
    """evaluate accuracy of model

    Args:
        model_teacher(keras model): teacher model
        model_student(keras model): student model
        data_loader(MNISTLoader object): dataloader for MNIST
        batch_size(int): batch size
    
    Returns:
        acc_teacher(numpy float): accuracy of teacher model
        acc_student(numpy float): accuracy of student model
    """
    accuracy_teacher = tf.keras.metrics.CategoricalAccuracy()
    accuracy_student = tf.keras.metrics.CategoricalAccuracy()
    data_generator = data_loader.get_batch(batch_size, training=False)
    num_batches = data_loader.num_test_data // batch_size
    for batch_index in range(num_batches):
        X, y = next(data_generator)

        y_pred_teacher = tf.nn.softmax(teacher(X))
        accuracy_teacher.update_state(y, y_pred_teacher)

        y_pred_student = tf.nn.softmax(student(X))
        accuracy_student.update_state(y, y_pred_student)
    acc_teacher = accuracy_teacher.result().numpy()
    acc_student = accuracy_student.result().numpy()
    logger = logging.getLogger(__name__)
    logger.critical("Teacher Acc: %f" % acc_teacher)
    logger.critical("Student Acc: %f" % acc_student)
    return acc_teacher, acc_student


def evaluate_backdoor_l2(model_backdoor):
    """L2-norm of backdoor mask

    Args:
        model_backdoor(keras model): backdoor model
    
    Returns:
        l2_norm(numpy float): l2_norm of backdoor mask
    """
    l2_norm = tf.norm(model_backdoor.get_mask(), ord='euclidean').numpy()
    logger = logging.getLogger(__name__)
    logger.critical("Backdoor l2 norm: %f" % l2_norm)
    return l2_norm


def evaluate_attack_success(model_teacher, model_student, data_loader, batch_size=128):
    """evaluate attack success rate

    Args:
        model_teacher(keras model): teacher model
        model_student(keras model): student model
        data_loader(MNISTLoader object): dataloader for MNIST
        batch_size(int): batch size
    
    Returns:
        succ_teacher(numpy float): attack success rate against teacher model
        succ_student(numpy float): attack success rate against student model
    """
    accuracy_teacher = tf.keras.metrics.CategoricalAccuracy()
    accuracy_student = tf.keras.metrics.CategoricalAccuracy()
    data_generator = data_loader.get_batch(batch_size, training=False)
    num_batches = data_loader.num_test_data // batch_size
    for batch_index in range(num_batches):
        X, y = next(data_generator)
        backdoored_X = backdoor(X)
        y_target = tf.one_hot(target_label, depth=10).numpy()
        y_target = np.tile(y_target, (batch_size, 1))

        y_pred_teacher = teacher(backdoored_X)
        accuracy_teacher.update_state(y_target, y_pred_teacher)

        y_pred_student = student(backdoored_X)
        accuracy_student.update_state(y_target, y_pred_student)
    succ_teacher = accuracy_teacher.result().numpy()
    succ_student = accuracy_student.result().numpy()
    logger = logging.getLogger(__name__)
    logger.critical("Attack success rate on Teacher: %f" % succ_teacher)
    logger.critical("Attack success rate on Student: %f" % succ_student)
    return succ_teacher, succ_student


# hyper parameters
num_epochs = 20
batch_size = 128
learning_rate_teacher = 1e-3
learning_rate_student = 1e-3
learning_rate_trigger = 1e-3
temperature = 8
target_label = 3

# load dataset
data_loader = MNISTLoader()
data_loader.preprocess()
data_generator = data_loader.get_batch(batch_size, training=True)

# build models
teacher = Teacher_model()
student = Student_model()
backdoor = Backdoor(target_label=target_label)

# optimizers
optimizer_teacher = tf.keras.optimizers.Adam(learning_rate=learning_rate_teacher)
optimizer_student = tf.keras.optimizers.Adam(learning_rate=learning_rate_student)
optimizer_backdoor = tf.keras.optimizers.Adam(learning_rate=learning_rate_trigger)

# customize training
for epoch_index in range(num_epochs):
    logger.critical("epoch: %d" % (epoch_index + 1))
    num_batches = data_loader.num_train_data // batch_size
    for batch_index in tqdm(range(num_batches), ascii=True):
        X, y = next(data_generator)
        y_target = tf.one_hot(target_label, depth=10).numpy()
        y_target = np.tile(y_target, (batch_size, 1))

        # train teacher
        with tf.GradientTape() as tape:
            logits_from_benign = teacher(X)
            backdoored_X = backdoor(X)
            logits_from_backdoor = teacher(backdoored_X)
            loss_teacher = loss_teacher_fn(
                logits_from_benign,
                logits_from_backdoor,
                y, y_target, temperature
            )
        grads = tape.gradient(loss_teacher, teacher.trainable_weights)
        optimizer_teacher.apply_gradients(
            grads_and_vars=zip(grads, teacher.trainable_weights)
        )

        # train student
        if batch_index % 2:
            with tf.GradientTape() as tape:
                logits_from_teacher = teacher(X)
                logits_from_student = student(X)
                loss_student = loss_student_fn(
                    logits_from_student,
                    logits_from_teacher,
                    y, temperature
                )
            grads = tape.gradient(loss_student, student.trainable_weights)
            optimizer_student.apply_gradients(
                grads_and_vars=zip(grads, student.trainable_weights)
            )
        
        # train backdoor
        with tf.GradientTape() as tape:
            loss_backdoor = loss_backdoor_fn(
                teacher, student, backdoor,
                X, y_target
            )
        grads = tape.gradient(loss_backdoor, backdoor.trainable_weights)
        optimizer_backdoor.apply_gradients(
            grads_and_vars=zip(grads, backdoor.trainable_weights)
        )
    evaluate_model_acc(teacher, student, data_loader, batch_size)
    evaluate_backdoor_l2(backdoor)
    evaluate_attack_success(teacher, student, data_loader, batch_size)