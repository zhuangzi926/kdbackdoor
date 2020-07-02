from . import losses
from . import metrics
from . import dynamic
from . import utils
from . import static

import os


def save_models(model_dir, cur_time, models):
    """save teacher, student, backdoor models

    save path pattern: {model_dir}/{cur_time}_{model_name}.h5
    """
    filename = os.path.join(model_dir, "{}_{}.h5")
    models["teacher"].save_weights(filename.format(cur_time, "teacher"))
    models["student"].save_weights(filename.format(cur_time, "student"))
    models["backdoor"].save_weights(filename.format(cur_time, "backdoor"))


def load_models(model_dir, cur_time, models):
    filename = os.path.join(model_dir, "{}_{}.h5")
    models["teacher"].load_weights(filename.format(cur_time, "teacher"))
    models["student"].load_weights(filename.format(cur_time, "student"))
    models["backdoor"].load_weights(filename.format(cur_time, "backdoor"))


def save_model(model_dir, cur_time, model, name):
    """save a single model"""
    filename = os.path.join(model_dir, "{}_{}.h5")
    model.save_weights(filename.format(cur_time, name))


def load_model(model_dir, cur_time, model, name):
    filename = os.path.join(model_dir, "{}_{}.h5")
    model.load_weights(filename.format(cur_time, name))
