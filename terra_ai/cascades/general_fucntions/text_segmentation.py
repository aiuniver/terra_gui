import numpy as np


def probability2classes(**params):

    classes = np.array(params['classes_names'])
    alpha = params['alpha']

    def fun(array):
        while len(array.shape) > 2:
            array = array[0]

        out = []
        for i in array:
            out.append(list(classes[i > alpha]))
        return out

    return fun


def main(**params):

    def fun(out):

        final = out[0]

        for i in out[1:]:
            new_final = np.zeros((final.shape[0] + params['step'], final.shape[1]))
            new_final[:-params['step']] += final
            final = new_final

            last = final[-params['length']:]
            last = (last + i) / 2
            final[-params['length']:] = last

        return np.array(final)

    return fun
