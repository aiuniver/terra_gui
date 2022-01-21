import os
import re
import pymorphy2
import numpy as np

from typing import Any
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from terra_ai.data.datasets.extra import LayerPrepareMethodChoice, LayerTextModeChoice
from terra_ai.utils import autodetect_encoding
from .base import Array


class TextArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        text_list = []
        txt_dict: dict = {}
        text: dict = {}
        open_tags, close_tags = options.get('open_tags'), options.get('close_tags')
        open_symbol, close_symbol = None, None
        if options.get('open_tags'):
            open_tags, close_tags = options['open_tags'].split(' '), options['close_tags'].split(' ')
            open_symbol = open_tags[0][0]
            close_symbol = close_tags[0][-1]
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['max_words']

        for idx, text_row in enumerate(sources):
            if os.path.isfile(str(text_row)):
                text_file = self.read_text(file_path=text_row, op_symbol=open_symbol, cl_symbol=close_symbol)
                if text_file:
                    txt_dict[text_row] = text_file
            else:
                if not text_row:
                    txt_dict[idx] = "nan"
                elif not isinstance(text_row, str):
                    txt_dict[idx] = str(text_row)
                else:
                    txt_dict[idx] = text_row

        if open_symbol:
            for key in txt_dict.keys():
                words = []
                for word in txt_dict[key].split(' '):
                    if word not in open_tags + close_tags:
                        words.append(word)
                txt_dict[key] = ' '.join(words)

        if options['pymorphy']:
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_dict.items():
                txt_dict[key] = self.apply_pymorphy(value, pymorphy)

        for key, value in sorted(txt_dict.items()):
            value = value.split(' ')
            if options['text_mode'] == 'completely':
                iter_count = 0
                adjust_flag = False
                adjusted_length = length
                while not adjust_flag:
                    adjust_length = length - len(
                        text_to_word_sequence(' '.join(value[0: adjusted_length]), options['filters'], lower=False))
                    adjusted_length += adjust_length
                    if adjust_length == 0 or iter_count == 10:
                        adjust_flag = True
                    iter_count += 1
                text[';'.join([str(key), f'[0-{adjusted_length}]'])] = ' '.join(value[0: adjusted_length])

            elif options['text_mode'] == 'length_and_step':
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    adjusted_length = length
                    if cur_step + length < len(value):
                        iter_count = 0
                        adjust_flag = False
                        while not adjust_flag:
                            adjust_length = length - len(
                                text_to_word_sequence(' '.join(value[cur_step: cur_step + adjusted_length]),
                                                      options['filters'], lower=False))
                            adjusted_length += adjust_length
                            if adjust_length == 0 or iter_count == 10:
                                adjust_flag = True
                            iter_count += 1
                    else:
                        stop_flag = True
                    text[';'.join([str(key), f'[{cur_step}-{cur_step + adjusted_length}]'])] = ' '.join(
                        value[cur_step: cur_step + adjusted_length])
                    cur_step += options['step'] + (adjusted_length - length)

        for elem in sorted(text.keys()):
            text_list.append(text[elem])

        instructions = {'instructions': text_list,
                        'parameters': {'prepare_method': options['prepare_method'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'text_mode': options['text_mode'],
                                       'length': options['length'],
                                       'max_words_count': options['max_words_count'],
                                       'word_to_vec_size': options['word_to_vec_size'],
                                       'filters': options['filters']
                                       }
                        }

        return instructions

    def create(self, source: Any, **options):

        instructions = {'instructions': source,
                        'parameters': options}

        return instructions

    def preprocess(self, source: Any, **options):

        array = []
        text = text_to_word_sequence(source, filters=options['filters'], lower=False, split=' ')
        words_to_add = []

        if options['prepare_method'] == LayerPrepareMethodChoice.embedding:
            array = options['preprocess'].texts_to_sequences([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.bag_of_words:
            array = options['preprocess'].texts_to_matrix([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
            for word in text:
                try:
                    array.append(options['preprocess'].wv[word])
                except KeyError:
                    array.append(np.zeros((options['length'],)))

        if len(array) < options['length']:
            if options['prepare_method'] in [LayerPrepareMethodChoice.embedding, LayerPrepareMethodChoice.bag_of_words]:
                words_to_add = [0 for _ in range((options['length']) - len(array))]
            elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                words_to_add = [[0 for _ in range(options['word_to_vec_size'])] for _ in
                                range((options['length']) - len(array))]
            array += words_to_add
        elif len(array) > options['length']:
            array = array[:options['length']]

        array = np.array(array)
        print("ARRAY: ", array)
        return array

    @staticmethod
    def read_text(file_path, op_symbol=None, cl_symbol=None) -> str:

        cur_text = autodetect_encoding(file_path)

        if op_symbol:
            cur_text = re.sub(op_symbol, f" {op_symbol}", cur_text)
            cur_text = re.sub(cl_symbol, f"{cl_symbol} ", cur_text)

        cur_text = ' '.join(text_to_word_sequence(
            cur_text, **{'lower': False, 'filters': '\r\t\n\ufeff\xa0', 'split': ' '})
        )

        return cur_text

    @staticmethod
    def apply_pymorphy(text, morphy) -> str:

        words_list = text.split(' ')
        words_list = [morphy.parse(w)[0].normal_form for w in words_list]

        return ' '.join(words_list)
