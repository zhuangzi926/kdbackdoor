import tensorflow as tf
import numpy as np

import settings


@tf.function
def loss_fn(
    models, backdoored_x, target_label, l2_factor=settings.BACKDOOR_L2_FACTOR,
):
    """loss function of backdoor model


	loss_student = softmax_with_logits(teacher(backdoor(X)), target)
		+ softmax_with_logits(student(backdoor(X)), target)
		+ L2_norm(mask_matrix)
	
	Args:
		models(Python dict): teacher, student, backdoor models
		x: a tf tensor of data, size = (batch_size, H, W, C)
		target_label: a tf tensor of target label, one-hot encoded, size = (batch_size, class_num)
	
	Returns:
		loss_backdoor: a tf tensor indicates loss of backdoor model
	"""
    logits_from_teacher = models["teacher"](backdoored_x)
    logits_from_student = models["student"](backdoored_x)
    loss_backdoor = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_label, logits=logits_from_teacher
    )
    loss_backdoor += tf.nn.softmax_cross_entropy_with_logits(
        labels=target_label, logits=logits_from_student
    )
    loss_backdoor += (
        tf.nn.l2_loss(models["backdoor"].get_mask() * models["backdoor"].get_trigger())
        * l2_factor
    )

    return tf.math.reduce_mean(loss_backdoor)


