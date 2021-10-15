import numpy as np
from tensorflow.keras.utils import to_categorical

from .array import min_max_scale


from ..common import decamelize
from .. import general_fucntions

import sys


def _classification(**params):

    classes_names = params['classes_names']

    def fun(x):
        index = []

        if params['type_processing'] == 'categorical':
            index = [classes_names.index(i) for i in x]
            index = to_categorical(index, num_classes=params['num_classes'], dtype='uint8')

        index = np.array(index)

        return index

    return fun


def _scaler(**params):
    min_max = min_max_scale(params['min_scaler'], params['max_scaler']) \
        if params['scaler'] == 'min_max_scaler' else None

    def fun(x):
        x = np.array(x)
        if min_max:
            x = min_max(x)

        return x

    return fun


def main(**params):
    process = []

    dataset_path = params['dataset_path']

    for key, i in params['columns'].items():
        try:
            process.append(getattr(sys.modules.get(__name__), '_' + decamelize(i['task']))(**i))
        except:
            type_module = getattr(general_fucntions, decamelize(i['task']))
            process.append(
                getattr(type_module, 'main')(**i, dataset_path=dataset_path, key=key)
            )
    columns = list(params['columns'].keys())

    def fun(data):
        out = np.zeros((data.shape[0], params['shape'][0]))
        j = 0
        for column, proc in zip(columns, process):
            i = data[column[2:]].to_numpy()

            if proc:
                x = proc(i)
                if len(x.shape) == 1:
                    out[:, j] = x
                    j += 1
                else:
                    out[:, j:j + x.shape[1]] = x
                    j += x.shape[1]
            else:
                out[:, j] = i
                j += 1

        return out

    return fun
