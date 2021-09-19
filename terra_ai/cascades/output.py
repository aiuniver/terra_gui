import cv2
import tensorflow
import numpy as np


def video(path, **params):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (params['width'], params['height'])
    )

    def fix(img):
        img = tensorflow.image.resize(img, (params['height'], params['width'])).numpy()
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
