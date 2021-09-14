import inspect
import json
import os
from pathlib import Path

from terra_ai import general_fucntions
from terra_ai.cascades.cascade import BuildModelCascade
from terra_ai.utils import decamelize

from tensorflow.keras.models import load_model
import tensorflow as tf


ROOT_PATH = str(Path(__file__).parent.parent)


def load_images(path):
    data = os.listdir(path)

    def fun():
        for image_path in data:
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_path))
            image = tf.keras.preprocessing.image.img_to_array(image)
            yield image

    return fun


def get_functions(preprocess):
    filter_func = lambda x: not x.startswith('_') and inspect.isfunction(getattr(preprocess, x))
    # {NAME_PREPROCESS: FUNCTION}
    functions = {name: getattr(preprocess, name) for name in dir(preprocess) if filter_func(name)}

    return functions


def list2tuple(inp: dict):
    for key, value in inp.items():
        if isinstance(value, list):
            inp[key] = tuple(value)

    return inp


def make_preprocess(preprocess_list):
    def fun(*x):
        out = []

        for i, (prep, element) in enumerate(zip(preprocess_list, x)):
            out.append(prep(element))

        return out

    return fun


def json2cascade(path: str):
    with open(path) as cfg:
        config = json.load(cfg)

    path_model = os.path.join(ROOT_PATH, config['weight'])

    model = load_model(path_model, compile=False, custom_objects=None)
    model.load_weights(os.path.join(path_model, config['model_name'] + '_best.h5'))

    if config['inputs']:
        preprocess = []

        for inp, param in config['inputs'].items():
            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(**param))

        preprocess = make_preprocess(preprocess)
    else:
        preprocess = None

    if config['outputs']:  # пока так
        postprocessing = None
        pass
    else:
        postprocessing = None

    model = BuildModelCascade(preprocess, model, postprocessing)

    return model
