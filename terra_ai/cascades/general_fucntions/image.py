import cv2
import numpy as np
from .array import change_type, change_size, min_max_scale


# def gaussian_blur(params: dict):
#     fun = lambda img: cv2.GaussianBlur(img, **params)
#
#     return fun


def main(**params):

    resize = change_size(params['shape']) if 'shape' in params.keys() else None
    retype = change_type(getattr(np, params['dtype'])) if 'dtype' in params.keys() else None
    min_max = min_max_scale(params['dataset_path'], params['key']) \
        if params['scaler'] == 'min_max_scaler' else None

    def fun(img):

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]
        if resize:
            img = resize(img)
        if min_max:
            img = min_max(img)
        if retype:
            img = retype(img)

        return img

    return fun
