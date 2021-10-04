import cv2
import tensorflow
import numpy as np


def change_size(shape: tuple):
    if len(shape) == 3:
        shape = shape[:2]

    fun = lambda frame: tensorflow.image.resize(frame, shape).numpy()

    return fun


def change_type(type):

    fun = lambda frame: frame.astype(type)

    return fun


def gaussian_blur(params: dict):
    fun = lambda img: cv2.GaussianBlur(img, **params)

    return fun


def main(**params):

    resize = change_size(params['shape'][-3:]) if 'shape' in params.keys() else None
    retype = change_type(getattr(np, params['dtype'])) if 'dtype' in params.keys() else None

    def fun(path):
        print(path)

        cap = cv2.VideoCapture(path)

        out = []
        array = []

        while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[:, :, [2, 1, 0]]

            if params['video_mode'] == 'completely' and params['max_frames'] > len(array) or\
               params['video_mode'] == 'length_and_step' and params['length'] > len(array):

                frame = resize(frame) if resize else frame
                frame = retype(frame) if retype else frame

                array.append(frame)

            elif params['video_mode'] == 'length_and_step':

                out.append(array)
                array = []

            else:
                break

        array = np.array(array)

        if array.shape[0] < params['max_frames']:
            if params['fill_mode'] == 'average_value':
                mean = np.mean(array, axis=0, dtype='uint16')
                array = np.concatenate(
                    (array, np.full((params['max_frames'] - array.shape[0], *mean.shape), mean, dtype='uint8'))
                )
                out.append(array)
        else:
            out.append(array)

        out = np.array(out)

        return out

    return fun
