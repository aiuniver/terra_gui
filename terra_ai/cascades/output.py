import cv2
import tensorflow
import numpy as np


def video(path, **params):

    shape = (params['width'], params['height'])

    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 10, shape
    )

    def fix(img):
        img = tensorflow.image.resize(img, shape[::-1]).numpy()
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def fun(img):
        if len(img.shape) == 4:
            for i in img:
                writer.write(fix(i))
        else:
            writer.write(fix(img))

    return writer, fun


def image(path):

    def fun(img):

        if len(img.shape) == 4:
            img = img[0]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path, img)

    return fun
