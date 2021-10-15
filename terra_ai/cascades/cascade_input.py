import cv2
import numpy as np
import os
from tensorflow.keras.utils import load_img
import pandas as pd


def video(paths):

    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        yield path


def image(path):
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


def audio(paths):
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        yield path


def dataframe(paths):
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        yield pd.read_csv(path)
