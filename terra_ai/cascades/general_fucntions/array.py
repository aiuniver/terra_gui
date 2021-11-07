import tensorflow
import joblib
import os


def change_type(type):
    fun = lambda frame: frame.astype(type)

    return fun


def change_size(shape: tuple):
    if len(shape) == 3:
        shape = shape[:2]

    fun = lambda frame: tensorflow.image.resize(frame, shape).numpy()

    return fun


def min_max_scale(dataset_path, key):
    preprocessing = joblib.load(
        os.path.join(
            dataset_path, 'preprocessing', key.split('_')[0], f'{key}.gz'
        )
    )

    def fun(x):
        orig_shape = x.shape
        x = preprocessing.transform(x.reshape(-1, 1))
        x = x.reshape(orig_shape).astype('float32')
        return x

    return fun
