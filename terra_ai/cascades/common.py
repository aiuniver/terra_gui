import inspect
import os
from pathlib import Path
import re

import tensorflow as tf


ROOT_PATH = str(Path(__file__).parent.parent.parent)

make_path = lambda path: os.path.join(ROOT_PATH, path)

type2str = lambda type: repr(type)[8:-2]


def decamelize(camel_case_string: str):
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    return re.sub("([a-z0-9])([A-Z0-9])", r"\1_\2", string).lower()


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


def yolo_decode(conv_output, num_class, i=0):
    strides = [8, 16, 32]

    anchors = [[[10, 13], [16, 30], [33, 23]],
               [[30, 61], [62, 45], [59, 119]],
               [[116, 90], [156, 198], [373, 326]]]
    # Train options
    # where i = 0, 1 or 2 to correspond to the three grid scales
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, num_class), axis=-1)

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
