import os
import time, datetime
import logging
import sys
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import datasets
import train
import nets
import settings


def config_args(args):
    settings.DEVICE = args.gpu
    settings.LR_BACKDOOR = args.lrbackdoor
    settings.BACKDOOR_L2_FACTOR = args.l2factor
    settings.SOFT_LABEL_RATE = args.soft_label_rate
    settings.TEMPERATURE = args.temperature


def config_gpu(args):
    # Designate gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def config_paths():
    # Configure output path
    root_dir = os.getcwd()
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

    dataset_name = "CIFAR10"
    teacher_name = "MobileNetV2"
    student_name = "CNN8"

    # Log dir
    log_dir = os.path.join(root_dir, "logs")
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)

    # Model dir
    model_dir = os.path.join(root_dir, "./models")
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    return root_dir, cur_time, log_dir, model_dir


def config_logger(cur_time, log_dir):
    # Set tf verbosity
    tf.get_logger().setLevel("INFO")

    # Logging setting
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(
        filename=os.path.join(log_dir, "%s.log" % cur_time), mode="a", encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def eval(models, dataset):
    """evaluate loss, acc, l2-norm of models on dataset"""
    logger = logging.getLogger(__name__)

    # Evaluate teacher
    loss = train.metrics.eval_model.loss(models["teacher"], dataset)
    acc = train.metrics.eval_model.acc(models["teacher"], dataset)
    logger.info("teacher test loss: {:.6f}".format(loss))
    logger.info("teacher test acc: {:.4f}".format(acc))

    # Evaluate student
    loss = train.metrics.eval_model.loss(models["student"], dataset)
    acc = train.metrics.eval_model.acc(models["student"], dataset)
    logger.info("student test loss: {:.6f}".format(loss))
    logger.info("student test acc: {:.4f}".format(acc))

    # Evaluate backdoor
    l2_norm = train.metrics.eval_backdoor.l2(models["backdoor"])
    logger.info("backdoor l2-norm: {:.6f}".format(l2_norm))

    # Evaluate attack
    success_rate = train.metrics.eval_attack.succ(
        models["teacher"], models["backdoor"], dataset
    )
    logger.info(
        "backdoor attack success rate against teacher: {:.4f}".format(success_rate)
    )
    success_rate = train.metrics.eval_attack.succ(
        models["student"], models["backdoor"], dataset
    )
    logger.info(
        "backdoor attack success rate against student: {:.4f}".format(success_rate)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=settings.DEVICE,
        required=False,
        help="select which gpu to use",
        dest="gpu",
    )
    parser.add_argument(
        "--lr_backdoor",
        type=float,
        default=settings.LR_BACKDOOR,
        required=False,
        help="manually set lr for backdoor",
        dest="lrbackdoor",
    )
    parser.add_argument(
        "--l2_factor",
        type=float,
        default=settings.BACKDOOR_L2_FACTOR,
        required=False,
        help="manually set l2 regularization factor for backdoor",
        dest="l2factor",
    )
    parser.add_argument(
        "--dynamic",
        help="whether use dynamic training or static training",
        dest="dynamic",
        action="store_true",
    )
    parser.add_argument(
        "--static",
        help="whether use dynamic training or static training",
        dest="dynamic",
        action="store_false",
    )
    parser.add_argument(
        "--soft_label_rate",
        type=float,
        default=settings.SOFT_LABEL_RATE,
        required=False,
        help="manually set soft label rate for distillation",
        dest="soft_label_rate",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=settings.TEMPERATURE,
        required=False,
        help="manually set temperature for distillation",
        dest="temperature",
    )
    args = parser.parse_args()
    config_args(args)
    config_gpu(args)

    root_dir, cur_time, log_dir, model_dir = config_paths()
    config_logger(cur_time, log_dir)

    logger = logging.getLogger(__name__)
    logger.info("Current root dir: {}".format(root_dir))
    logger.info("Current log dir: {}".format(log_dir))
    logger.info("Current log file name: {}.log".format(cur_time))

    logger.debug("Loading data...")
    data_loader = datasets.gtsrb.Loader()
    data_loader.preprocess(
        func_train=datasets.utils.convert, func_test=datasets.utils.convert
    )

    dataset_train = data_loader.get_dataset(training=True)
    dataset_test = data_loader.get_dataset(training=False)

    logger.debug("Building models...")
    models = train.utils.build_models()
    optimizers = train.utils.get_opts()

    if args.dynamic is True:
        logger.debug("Pretrain teacher...")
        pretrain_loss, pretrain_acc = train.dynamic.pretrain(
            models["teacher"], dataset_train, dataset_test
        )
        logger.info(
            "teacher pretrain loss: {:.6f}, teacher pretrain acc: {:.6f}".format(
                pretrain_loss, pretrain_acc
            )
        )

        logger.debug("Starting dynamic distillation")
        for epoch_index in range(settings.NUM_EPOCHS):
            logger.info("epoch: %d" % (epoch_index + 1))

            (
                kd_loss_teacher,
                kd_loss_student,
                kd_loss_backdoor,
            ) = train.dynamic.train_epoch(models, dataset_train, optimizers,)

            logger.info("teacher kd loss: {}".format(kd_loss_teacher.numpy()))
            logger.info("student kd loss: {}".format(kd_loss_student.numpy()))
            logger.info("backdoor kd loss: {}".format(kd_loss_backdoor.numpy()))

            eval(models, dataset_test)
        train.save_models(model_dir, cur_time, models)

    else:
        train.load_model(model_dir, "2020-08-03-2247", models["teacher"], "teacher")
        train.load_model(model_dir, "2020-08-03-2247", models["backdoor"], "backdoor")
        models["student"] = nets.resnet_v1.get_model(depth=8)
        optimizers = train.utils.get_opts()
        logger.debug("Starting static distillation")
        for epoch_index in range(settings.NUM_EPOCHS):
            logger.info("static epoch: %d" % (epoch_index + 1))

            kd_loss_student = train.static.train_epoch(models, dataset_train, optimizers,)
            
            logger.info("student static kd loss: {}".format(kd_loss_student.numpy()))

            eval(models, dataset_test)
        train.save_model(
            model_dir, cur_time, models["student"], "student_static_resnet_v1_8"
        )
