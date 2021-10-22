import cv2
import numpy as np
from .array import change_type, change_size, min_max_scale


def main(**params):

    resize = change_size(params['shape'][-3:]) if 'shape' in params.keys() else None
    retype = change_type(getattr(np, params['dtype'])) if 'dtype' in params.keys() else None
    min_max = min_max_scale(params['dataset_path'], params['key']) \
        if params['scaler'] == 'min_max_scaler' else None

    def fun(path):

        cap = cv2.VideoCapture(path)

        out = []
        array = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = frame[:, :, [2, 1, 0]]

            if min_max:
                frame = min_max(frame)

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
