import tensorflow as tf
import numpy as np

import nets
import settings


def build_models():
    """return teacher, student, backdoor models"""
    teacher = nets.mobilenet_v2.get_model(
        settings.IMG_SHAPE, k=settings.NUM_CLASSES, alpha=1.0
    )
    student = nets.cnn8.get_model()
    backdoor = nets.backdoor.get_model()

    models = {
        "teacher": teacher,
        "student": student,
        "backdoor": backdoor,
    }

    return models


def get_lr_scheduler(model=None):
    """manually set lr

    Args:
        model(str): "teacher", "student", "backdoor"
    
    Rets:
        A lr scheduler object for teacher or student, a constant for backdoor
    """
    if model == "teacher":
        lr = settings.LR_TEACHER
    elif model == "student":
        lr = settings.LR_STUDENT
    else:
        return settings.LR_BACKDOOR

    steps_per_epoch = settings.NUM_TRAIN_DATA // settings.BATCH_SIZE
    step_boundaries = [e * steps_per_epoch for e in settings.EPOCH_BOUNDARIES]

    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(step_boundaries, lr)



def get_opts():
    """return teacher, student, backdoor opts"""
    opt_teacher = tf.keras.optimizers.SGD(
        learning_rate=get_lr_scheduler(model="teacher"), momentum=0.9
    )
    opt_student = tf.keras.optimizers.SGD(
        learning_rate=get_lr_scheduler(model="student"), momentum=0.9
    )
    opt_backdoor = tf.keras.optimizers.SGD(
        learning_rate=get_lr_scheduler(model="backdoor"), momentum=0.9
    )

    optimizers = {
        "teacher": opt_teacher,
        "student": opt_student,
        "backdoor": opt_backdoor,
    }

    return optimizers
