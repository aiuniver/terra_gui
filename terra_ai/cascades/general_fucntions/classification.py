import numpy as np


def main(**params):

    classes_names = params['classes_names'] if 'classes_names' in params.keys() else None

    def fun(acc):
        acc *= 100
        acc = np.round(np.array(acc), 2)
        acc = acc.astype(np.int)

        if len(acc.shape) == 2:
            acc = acc[0]
        acc = list(zip(classes_names, acc))

        return sorted(acc, key=lambda x: x[1], reverse=True)

    return fun
