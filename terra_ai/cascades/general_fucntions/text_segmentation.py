import numpy as np
import re


def put_tag(open_tag, close_tag, alpha=0.6):
    tags = {i: j for i, j in zip(open_tag.split(), close_tag.split())}
    classes = np.array(open_tag.split())

    def fun(text, array):

        CLEANR = re.compile('<.*?>')

        text = re.sub(CLEANR, '', text)

        array = np.array(array)
        while len(array.shape) > 2:
            array = array[0]

        indexes = []
        word = ''
        for index, ch in enumerate(text):
            if re.match('[0-9А-Яа-яЁёa-zA-Z]', ch):
                word += ch
            elif len(word):
                indexes.append(index - len(word))
                word = ''

        past_id = 0
        new_text = ''
        past_tag = []

        for i, j in zip(array, indexes):
            i = list(classes[i > alpha])
            for tag in i:
                if tag not in past_tag:
                    new_text += tag

            new_text += text[past_id: j]

            for tag in past_tag:
                if tag not in i:
                    new_text += tags[tag]

            past_tag = i
            past_id = j

        new_text += text[past_id:]

        return new_text

    return fun


# def probability2classes(**params):  пока не нужно
#     classes = np.array(params['classes_names'])
#     alpha = params['alpha']
#
#     def fun(array):
#         array = np.array(array)
#
#         while len(array.shape) > 2:
#             array = array[0]
#
#         out = []
#         for i in array:
#             out.append(list(classes[i > alpha]))
#         return out
#
#     return fun


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

        final = np.array(final)
        return final

    return fun
