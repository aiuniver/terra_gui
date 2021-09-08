import cv2


def resize_img(shape: tuple):
    fun = lambda frame: cv2.resize(frame, shape)

    return fun


def gaussian_blur(params: dict):
    fun = lambda img: cv2.GaussianBlur(img, **params)

    return fun
