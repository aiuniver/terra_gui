import numpy as np


def main(**params):

    classes_names = params['classes_names'] if 'classes_names' in params.keys() else None

    def fun(acc):
        acc *= 100
        acc = np.array(acc)
        # print(acc.shape)
        if len(acc) == 1:
            acc = acc[0]
        else:
            acc = np.mean(acc, axis=0)
        acc = acc.round().astype(np.int)
        out = list(zip(classes_names, acc))
        return sorted(out, key=lambda x: x[-1], reverse=True)

    return fun