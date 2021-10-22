import numpy as np
import joblib
import os


def _scaler(**params):

    preprocessing = joblib.load(
        os.path.join(
            params['dataset_path'], 'preprocessing', params["key"].split('_')[0], f'{params["key"]}.gz'
        )
    )

    def fun(x):
        orig_shape = x.shape
        x = preprocessing.inverse_transform(x.reshape(-1, 1))
        x = x.reshape(orig_shape).astype('float32')
        x = np.round(x, 2)

        return x

    return fun


def main(**params):
    dataset_path = params['dataset_path']

    process = []

    for key, i in params['columns'].items():
        process.append(
            _scaler(**i, dataset_path=dataset_path, key=key)
        )

    def fun(arr):
        arr = arr.numpy()

        for column, proc in enumerate(process):
            arr[:, :, column] = proc(arr[:, :, column])

        return arr

    return fun
