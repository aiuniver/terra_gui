import re

import numpy as np
import pymorphy2
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from terra_ai.data.datasets.extra import LayerTextModeChoice, LayerPrepareMethodChoice
from terra_ai.datasets.preprocessing import CreatePreprocessing


def main(**params):
    open_symbol = close_symbol = None

    if params['parameters']['open_tags']:
        open_symbol = params['parameters']['open_tags'].split(' ')[0][0]
        close_symbol = params['parameters']['close_tags'].split(' ')[0][-1]

    if open_symbol:
        tags = params['parameters']['close_tags'].split(' ') + params['parameters']['open_tags'].split(' ')

    length = \
        params['parameters']['length'] if\
            params['parameters']['text_mode'] == LayerTextModeChoice.length_and_step else \
            params['parameters']['max_words']

    preprocessing = CreatePreprocessing(params['dataset_path'])
    preprocessing.load_preprocesses("1")
    preprocessing = preprocessing.preprocessing["1"]

    def fun(text):
        if open_symbol:
            text = re.sub(open_symbol, f" {open_symbol}", text)
            text = re.sub(close_symbol, f"{close_symbol} ", text)

        text = ' '.join(text_to_word_sequence(text, filters=params['parameters']['filters']))

        if open_symbol:
            text = ' '.join([word for word in text.split() if word not in tags])

        if params['parameters']['pymorphy']:
            morphy = pymorphy2.MorphAnalyzer()
            text = ' '.join([morphy.parse(w)[0].normal_form for w in text.split(' ')])

        if params['parameters']['text_mode'] == LayerTextModeChoice.completely:
            text = ' '.join(text.split(' ')[:params['parameters']['max_words']])
        elif params['parameters']['text_mode'] == LayerTextModeChoice.length_and_step:
            max_length = len(text.split(' '))
            cur_step = 0
            stop_flag = False
            while not stop_flag:
                text = ' '.join(text.split(' ')[cur_step: cur_step + length])
                cur_step += params['parameters']['step']
                if cur_step + params['parameters']['length'] > max_length:
                    stop_flag = True

        array = []
        text = text.split(' ')
        words_to_add = []

        if params['parameters']['prepare_method'] == LayerPrepareMethodChoice.embedding:
            array = preprocessing['object_tokenizer'].texts_to_sequences([text])[0]
        elif params['parameters']['prepare_method'] == LayerPrepareMethodChoice.bag_of_words:
            array = preprocessing['object_tokenizer'].texts_to_matrix([text])[0]
        elif params['parameters']['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
            for word in text:
                try:
                    array.append(preprocessing['object_word2vec'][word])
                except KeyError:
                    array.append(np.zeros((params['parameters']['length'],)))

        if len(array) < params['parameters']['length']:
            if params['parameters']['prepare_method'] in \
                    [LayerPrepareMethodChoice.embedding, LayerPrepareMethodChoice.bag_of_words]:
                words_to_add = [0 for _ in range((params['parameters']['length']) - len(array))]
            elif params['parameters']['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                words_to_add = [[0 for _ in range(params['word_to_vec_size'])] for _ in
                                range((params['parameters']['length']) - len(array))]
            array += words_to_add

        array = np.array(array)
        array = array[np.newaxis, ...]

        return array

    return fun
