import cv2
import numpy as np
import os


def video(path):

    cap = cv2.VideoCapture(path)

    while cap.isOpened:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame


def image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, ...]
    return img


def folder(path):
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, ...]
        yield img


def text(path):
    with open(path, 'r', encoding='utf-8') as txt:
        text = txt.read()

    return text
