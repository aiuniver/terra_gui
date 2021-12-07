import re
import os

import numpy as np
import pymorphy2
import joblib
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def main(**params):
    open_tags, close_tags = None, None
    open_symbol, close_symbol = None, None

    if params.get('open_tags'):
        open_tags, close_tags = params['open_tags'].split(' '), params['close_tags'].split(' ')
    if open_tags:
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]

    length = params['length'] if params['text_mode'] == "length_and_step" else \
        params['max_words']
    preprocessing = joblib.load(
        os.path.join(
            params['dataset_path'], 'preprocessing', params["key"].split('_')[0], f'{params["key"]}.gz'
        )
    )

    def fun(text):
        if not isinstance(text, str):
            text = text[params['cols_names']][0]

        if open_symbol:
            text = re.sub(open_symbol, f" {open_symbol}", text)
            text = re.sub(close_symbol, f"{close_symbol} ", text)

        text = ' '.join(text_to_word_sequence(text, lower=True, filters=params['filters'], split=' '))

        if open_symbol:
            text = ' '.join([word for word in text.split() if word not in open_tags + close_tags])

        if params['pymorphy']:
            morphy = pymorphy2.MorphAnalyzer()
            text = ' '.join([morphy.parse(w)[0].normal_form for w in text.split(' ')])

        text = text.split()
        arr = []
        if params['text_mode'] == 'completely':
            text = text[:params['max_words']]
            arr.append(text)
        else:
            for i in range(0, len(text) - length + params['step'], params['step']):
                arr.append(text[i: i + length])
            if len(text) < length:
                arr.append(text)

        array = []

        if params['prepare_method'] == "embedding":
            array = preprocessing.texts_to_sequences(arr)
            for arr in array:
                for _ in range(length - len(arr)):
                    arr.append(0)
        elif params['prepare_method'] == "bag_of_words":
            array = preprocessing.texts_to_matrix(arr)
        elif params['prepare_method'] == "word_to_vec":
            for word in arr:
                try:
                    tmp_list = preprocessing[word]
                    tmp_arr = np.array(tmp_list)
                    if tmp_arr.shape[0] < length:
                        new_array = np.zeros((length, params['word_to_vec_size']))
                        new_array[:tmp_arr.shape[0], :] = tmp_arr[:, :]
                        array.append(new_array)
                    elif tmp_arr.shape[0] > length:
                        new_array = np.zeros((length, params['word_to_vec_size']))
                        new_array[:, :] = tmp_arr[:length, :]
                        array.append(new_array)

                except KeyError:
                    array.append(np.zeros((length, params['word_to_vec_size'])))
            array = np.array(array)

            # if array.shape[1] < length:
            #     new_array = np.zeros((1, length, params['word_to_vec_size']))
            #     new_array[:, :array.shape[1]] += array
            #     array = new_array
            # elif array.shape[1] > length:
            #     n = (array.shape[0] % length) + 1
            #     new_array = np.zeros((n, length, params['word_to_vec_size']))
            #     for i in range(n):
            #         new_array[i][:len(array[i])] += array[i]
            #
            #     array = new_array

        array = np.array(array)

        return array

    return fun
