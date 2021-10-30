import numpy as np


def _classification(**params):

    classes_names = params['classes_names']

    def fun(acc):
        s = sum(acc)
        acc /= s
        acc *= 100
        acc = np.array(acc)
        acc = acc.astype(np.int)

        out = list(zip(classes_names, acc))
        out = sorted(out, key=lambda x: x[-1], reverse=True)
        return out

    return fun


def main(**params):
    process = []
    for key, i in params['columns'].items():
        process.append(
            _classification(**i)
        )

    def fun(arr):
        arr = arr.numpy()

        out = []
        for column, proc in enumerate(process):
            out.append(proc(arr[column]))

        return out

    return fun
