import inspect
import json
import os
from pathlib import Path

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
