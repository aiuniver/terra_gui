import numpy as np
import joblib
import os


def main(**params):

    preprocessing = joblib.load(
        os.path.join(
            params['dataset_path'], 'preprocessing', params["key"].split('_')[0], f'{params["key"]}.gz'
        )
    )

    def fun(arr):
        arr = preprocessing.inverse_transform(arr)
        arr = np.round(arr, 2)
        arr = arr.reshape(-1)
        return arr

    return fun
