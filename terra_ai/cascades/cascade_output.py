import cv2
import tensorflow
import numpy as np
import os
from tensorflow.keras.utils import save_img
from pydub import AudioSegment


def video(path, **params):

    shape = (params['width'], params['height'])

    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"VP80"), 15, shape
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
        while len(img.shape) != 3:
            img = img[0]

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path, img)

        save_img(path, img)

    return fun


def text(path):
    if not os.path.isfile(path):
        with open(path, "w"):
            pass

    def fun(string):
        with open(path, 'a') as f:
            f.write(str(string) + '\n')

    return fun


def google_tts(path):
    def fun(tts):
        tts.save(path)

    return fun


def audio(path):

    def fun(iobytes):
        song = AudioSegment.from_file(iobytes, format="mp3")
        song.export(path, format="webm")

    return fun