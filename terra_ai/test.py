import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import json
import os
from pathlib import Path

from terra_ai.cascades.cascade import CascadeElement
from terra_ai.general_fucntions.image import preprocessing
from terra_ai.common import get_functions


ROOT_PATH = str(Path(__file__).parent.parent)


def load_images(path):
    data = os.listdir(path)

    def fun():
        for image_path in data:
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_path))
            image = np.array(image)

            yield image

    return fun


def parse_json(path: str):

    with open(path) as cfg:
        config = json.load(cfg)

    path_model = os.path.join(ROOT_PATH, config['weight'])

    print(path_model)
    model = load_model(path_model, compile=False, custom_objects=None)
    # model.load_weights(os.path.join(path_model, params['model_name'] + '_best.h5'))

    return config, model


config = parse_json(os.path.join(ROOT_PATH, "test_example/airplanes/config.json"))
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
