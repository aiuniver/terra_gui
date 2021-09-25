import cv2
import tensorflow
import numpy as np


def change_size(shape: tuple):
    if len(shape) == 3:
        shape = shape[:2]

    fun = lambda frame: tensorflow.image.resize(frame, shape).numpy()

    return fun


def change_type(type):

    fun = lambda frame: frame.astype(type)

    return fun


def gaussian_blur(params: dict):
    fun = lambda img: cv2.GaussianBlur(img, **params)

    return fun


def main(**params):

    resize = change_size(params['shape']) if 'shape' in params.keys() else None
    retype = change_type(getattr(np, params['dtype'])) if 'dtype' in params.keys() else None

    def fun(img):

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]

        if resize:
            img = resize(img)
        if retype:
            img = retype(img)

        return img

    return fun
