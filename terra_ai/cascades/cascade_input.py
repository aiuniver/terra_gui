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
    # img = cv2.imread(path)
    with open(path, "rb") as img_source:
        img = img_source.read()
    img_arr = np.frombuffer(img, dtype=np.uint8)
    out_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    out_img = out_img[np.newaxis, ...]
    return out_img


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
