import numpy as np


def main(**params):

    classes_names = params['classes_names'] if 'classes_names' in params.keys() else None

    def fun(acc):

        acc = np.round(np.array(acc), 3)

        return list(zip(classes_names, acc))

    return fun
