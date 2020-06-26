import tensorflow as tf
import numpy as np


def l2(backdoor):
    """L2-norm of backdoor mask

    Args:
        backdoor(tf.keras.Model)
    
    Returns:
        l2_norm(tensor)
    """
    l2_norm = tf.norm(backdoor.get_mask(), ord="euclidean")
    return l2_norm
