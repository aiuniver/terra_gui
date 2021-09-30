import cv2
import numpy as np
import os
from tensorflow.keras.utils import load_img


def video(path):

    cap = cv2.VideoCapture(path)

    while cap.isOpened:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame


def image(path):
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = load_img(path)
    out_img = np.array(img)
    out_img = out_img[np.newaxis, ...]
    return out_img


def folder(path):
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, ...]
        yield img


def text(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as txt:
            text = txt.read()
        yield text
