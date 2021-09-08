import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import json
import os
from pathlib import Path

from terra_ai.cascades.cascade import BuildModelCascade
from terra_ai.common import get_functions
from terra_ai import general_fucntions


ROOT_PATH = str(Path(__file__).parent.parent)


def load_images(path):
    data = os.listdir(path)

    def fun():
        for image_path in data:
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_path))
            image = np.array(image)

            yield image

    return fun


def json2cascade(path: str):

    with open(path) as cfg:
        config = json.load(cfg)

    path_model = os.path.join(ROOT_PATH, config['weight'])

    model = load_model(path_model, compile=False, custom_objects=None)
    model.load_weights(os.path.join(path_model, config['model_name'] + '_best.h5'))

    type_model = config['tags'][0]['alias']
    type_model_module = getattr(general_fucntions, type_model)

    if config['preprocess']:
        preprocess_functions = get_functions(getattr(type_model_module, 'preprocess'))

        with open(os.path.join(ROOT_PATH, config['preprocess'])) as cfg:
            preprocess = json.load(cfg)

        preprocess = [preprocess_functions[name](param) for name, param in preprocess.items()]
        preprocess = getattr(type_model_module, 'make_preprocess')(preprocess)
    else:
        preprocess = None

    if config['postprocessing']:  # пока так
        postprocessing = None
        pass
    else:
        postprocessing = None

    model = BuildModelCascade(preprocess, model, postprocessing)

    return model


config = json2cascade(os.path.join(ROOT_PATH, "test_example/airplanes/config.json"))
print(config)

# INPUT_TRDS_FRONT = load_images('/home/evgeniy/terra_gui/TerraProjects/airplanes.trds/sources/1_image/Самолеты')
#
#
# with open('/terra_ai/test_example/airplanes/preprocess.json') as cfg:
#     preprocess = json.load(cfg)
#
# # preprocess_list = []
# # for name, param in preprocess.items():
# #     preprocess_list.append(FUNCTIONS[name](param))
#
# FUNCTIONS = get_functions()
# preprocess_list = [FUNCTIONS[name](param) for name, param in preprocess.items()]
#
#
# preprocess = CascadeElement(preprocessing(preprocess_list), name="Препроцесс")
#
# # это делавет фронт
# for img in INPUT_TRDS_FRONT():
#     img = preprocess(img)
#     plt.imshow(img)
#     plt.show()
