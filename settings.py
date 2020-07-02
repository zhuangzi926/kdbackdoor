# GPU
DEVICE = 3

# Dataset params
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)

NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

DATASET_MEAN = CIFAR10_MEAN
DATASET_STD = CIFAR10_STD

# Train params
NUM_EPOCHS = 100
BATCH_SIZE = 128
NUM_PRETRAIN_EPOCHS = 10
PRETRAIN_LR = 0.1
EPOCH_BOUNDARIES = [80, 160]
# SGD_LR = [0.1, 0.01, 0.001]
LR_TEACHER = [1e-3, 1e-4, 1e-5]
LR_STUDENT = [1e-2, 1e-3, 1e-4]
LR_BACKDOOR = 1e-4

TEMPERATURE = 8

# Attack params
TARGET_LABEL = 3
TEACHER_POISONED_RATE = 0.01
BACKDOOR_L2_FACTOR = 0.05
