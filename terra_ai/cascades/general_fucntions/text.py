import re
import os

import numpy as np
import pymorphy2
import joblib
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from terra_ai.cascades.common import decamelize


def main(**params):
    open_tags, close_tags = None, None
    open_symbol, close_symbol = None, None

    if params.get('open_tags'):
        open_tags, close_tags = params['open_tags'].split(' '), params['close_tags'].split(' ')
    if open_tags:
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]

    length = params['length'] if params['text_mode'] == "length_and_step" else params['max_words']

    preprocessing = joblib.load(
        os.path.join(
            params['dataset_path'], 'preprocessing', str(params['key']),
            f'{params["key"]}_{decamelize(params["task"])}.gz')
    )
    print(os.path.join(
            params['dataset_path'], 'preprocessing', str(params['key']),
            f'{params["key"]}_{decamelize(params["task"])}.gz'))

    def fun(text):
        if open_symbol:
            text = re.sub(open_symbol, f" {open_symbol}", text)
            text = re.sub(close_symbol, f"{close_symbol} ", text)

        text = ' '.join(text_to_word_sequence(text, lower=True, filters=params['filters']))

        if open_symbol:
            text = ' '.join([word for word in text.split() if word not in open_tags + close_tags])

        if params['pymorphy']:
            morphy = pymorphy2.MorphAnalyzer()
            text = ' '.join([morphy.parse(w)[0].normal_form for w in text.split(' ')])

        text = text.split()

        if params['text_mode'] == "completely":
            arr = [text[:params['max_words']]]
        elif params['text_mode'] == "length_and_step":
            arr = []
            for i in range(0, len(text) - length + params['step'], params['step']):
                arr.append(text[i: i + length])

        array = []

        if params['prepare_method'] == "embedding":
            array = preprocessing.texts_to_sequences(arr)
            for arr in array:
                for _ in range(length - len(arr)):
                    arr.append(0)
        elif params['prepare_method'] == "bag_of_words":
            array = preprocessing.texts_to_matrix(arr)
        # elif params['prepare_method'] == "word_to_vec":
        #     print(preprocessing)
        #     for word in arr:
        #         try:
        #             array.append(preprocessing[word])
        #         except KeyError:
        #             array.append(np.zeros((params['parameters']['length'],)))

        # if len(array) < params['length']:
        #     if params['prepare_method'] in ["embedding", "bag_of_words"]:
        #         words_to_add = [0 for _ in range((params['parameters']['length']) - len(array))]
        #     elif params['prepare_method'] == "word_to_vec":
        #         words_to_add = [[0 for _ in range(params['word_to_vec_size'])] for _ in
        #                         range((params['length']) - len(array))]
        #     array += words_to_add

        array = np.array(array)

        return array

    return fun
