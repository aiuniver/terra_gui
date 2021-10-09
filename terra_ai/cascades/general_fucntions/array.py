import tensorflow


def change_type(type):
    fun = lambda frame: frame.astype(type)

    return fun


def change_size(shape: tuple):
    if len(shape) == 3:
        shape = shape[:2]

    fun = lambda frame: tensorflow.image.resize(frame, shape).numpy()

    return fun


def min_max_scale(max_scale, min_scale):
    diff = max_scale - min_scale

    fun = lambda img: (img - img.min()) / (img.max() - img.min()) * diff + min_scale

    return fun
