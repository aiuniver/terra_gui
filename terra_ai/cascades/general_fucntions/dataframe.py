import numpy as np
from tensorflow.keras.utils import to_categorical
import joblib
import os


from ..common import decamelize
from .. import general_fucntions

import sys


def _classification(**params):

    classes_names = params['classes_names']

    def fun(x):
        index = []

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        if params['type_processing'] == 'categorical':
            if x.shape:
                index = [classes_names.index(i) for i in x]
            else:
                index = [classes_names.index(x)]
            index = to_categorical(index, num_classes=params['num_classes'], dtype='uint8')

        index = np.array(index)

        return index

    return fun


def _scaler(**params):

    preprocessing = joblib.load(
        os.path.join(
            params['dataset_path'], 'preprocessing', params["key"].split('_')[0], f'{params["key"]}.gz'
        )
    )

    def fun(x):
        x = np.array(x)
        if x.shape and len(x.shape) == 1:
            x = x.reshape(-1, 1)
        else:
            x = np.array([[x]])
        x = preprocessing.transform(x)
        return x

    return fun


def main(**params):
    process = []

    dataset_path = params['dataset_path']

    for key, i in params['columns'].items():
        try:
            process.append(getattr(sys.modules.get(__name__), '_' + decamelize(i['task']))(
                **i, dataset_path=dataset_path, key=key
            ))
        except:
            type_module = getattr(general_fucntions, decamelize(i['task']))
            process.append(
                getattr(type_module, 'main')(**i, dataset_path=dataset_path, key=key)
            )
    columns = list(params['columns'].keys())

    def fun(data):
        out = []
        if len(params['shape']) == 1:
            if len(data.shape) == 1:
                out = np.zeros((1, params['shape'][0]))
            else:
                out = np.zeros((data.shape[0], params['shape'][0]))
        elif len(params['shape']) == 2:
            out = np.zeros((params['shape']))
        lenght = out.shape[0]

        j = 0
        for column, proc in zip(columns, process):

            i = np.array(data[column[2:]])

            if proc:
                x = proc(i[:lenght])
                if len(x.shape) == 1:
                    out[:, j] = x
                    j += 1
                else:
                    out[:, j:j + x.shape[1]] = x
                    j += x.shape[1]
            else:
                out[:, j] = i
                j += 1

        if len(params['shape']) == 2:
            out = out[np.newaxis, ...]

        return out

    return fun
