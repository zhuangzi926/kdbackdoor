import tensorflow as tf
import numpy as np

import settings


@tf.function
def loss_fn(
    logits_from_student,
    logits_from_teacher,
    benign_label,
    temperature,
    soft_label_rate=settings.SOFT_LABEL_RATE,
):
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
    loss_student = (
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_label_from_teacher, logits=logits_from_student / temperature
        )
        * soft_label_rate
    )
    loss_student += (
        tf.nn.softmax_cross_entropy_with_logits(
            labels=benign_label, logits=logits_from_student
        )
        * (1 - soft_label_rate)
    )
    return tf.math.reduce_mean(loss_student)
