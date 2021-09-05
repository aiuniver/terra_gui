import cv2
import numpy as np
from typing import Callable


def count(bbox: np.ndarray) -> int:
    return bbox.shape[-2]


def head_cropping(width_object: int = 80, out_size: int = 64) -> Callable:
    size = (out_size, out_size)
    resize_img = lambda x: cv2.resize(x, size) / 255

    def fun(bbox: np.ndarray, img: np.ndarray) -> np.ndarray:
        heads = []

        for b in bbox:
            center = (b[3] + b[1]) // 2
            top = 0 if center < width_object else center - width_object
            bot = center + width_object

            center = (b[2] + b[0]) // 2
            left = 0 if center < width_object else center - width_object
            right = center + width_object

            heads.append(
                resize_img(img[top: bot, left: right])
            )

        return np.array(heads)

    return fun
