import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import json
import os
import inspect

from terra_ai import general_preprocess
from terra_ai.cascades.cascade import CascadeElement


def load_images(path):
    data = os.listdir(path)

    def fun():
        for image_path in data:
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_path))
            image = np.array(image)

            yield image

    return fun


INPUT_TRDS_FRONT = load_images('/home/evgeniy/terra_gui/TerraProjects/airplanes.trds/sources/1_image/Самолеты')


filter_func = lambda x: not x.startswith('_') and inspect.isfunction(getattr(general_preprocess, x))

# {NAME_PREPROCESS: FUNCTION}
FUNCTIONS = {name: getattr(general_preprocess, name) for name in dir(general_preprocess) if filter_func(name)}

with open('/home/evgeniy/terra_gui/terra_ai/cascades/preprocess.json') as cfg:
    preprocess = json.load(cfg)

# preprocess_list = []
# for name, param in preprocess.items():
#     preprocess_list.append(FUNCTIONS[name](param))

preprocess_list = [FUNCTIONS[name](param) for name, param in preprocess.items()]


def preprocessing(preprocess_list: list):
    def fun(x):
        for prep in preprocess_list:
            x = prep(x)

    return fun


preprocess = CascadeElement(preprocessing(preprocess_list), name="Preprocess")


# это делавет фронт
for img in INPUT_TRDS_FRONT():
    img = preprocess(img)
    plt.imshow(img)
    plt.show()
