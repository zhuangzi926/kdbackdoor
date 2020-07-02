import tensorflow as tf
import numpy as np

import settings


@tf.function
def loss_fn(
    logits_from_benign,
    logits_from_backdoor,
    benign_label,
    target_label,
    temperature,
    poisoned_rate=settings.TEACHER_POISONED_RATE,
):
    """loss function of teacher model

    loss_teacher = softmax_with_logits(teacher(X) / T, y) * (1 - poisoned_rate)
    + softmax(teacher(X_t) / T, target) * poisoned_rate

    Args:
        logits_from_benign: a tensor from output of teacher model, size = (batch_size, class_num)
        logits_from_backdoor: a tensor from backdoored output of teacher model, size = (batch_size, class_num)
        benign_label: a numpy array from dataset, one-hot encoded, size = (batch_size, class_num)
        target_label: a numpy array of target label, one-hot encoded, size = (batch_size, class_num)
        temperature: an int, hyperparameter that controls knowledge distillation
        poisoned_rate: a float, indicating how much teacher is affected by backdoored data
    
    Returns:
        loss_teacher: a float value indicates loss of teacher model
    """

    loss_teacher = tf.nn.softmax_cross_entropy_with_logits(
        labels=benign_label, logits=logits_from_benign / temperature
    ) * (1 - poisoned_rate)
    loss_teacher += (
        tf.nn.softmax_cross_entropy_with_logits(
            labels=target_label, logits=logits_from_backdoor / temperature
        )
        * poisoned_rate
    )

    return tf.math.reduce_mean(loss_teacher)
