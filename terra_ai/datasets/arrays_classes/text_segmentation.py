import re
import numpy as np

from ast import literal_eval
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from typing import Any

from terra_ai.data.datasets.extra import LayerTextModeChoice
from terra_ai.utils import autodetect_encoding
from .base import Array


class TextSegmentationArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        """

                Args:
                    sources: list
                        Пути к файлам.
                    **options:
                        open_tags: str
                            Открывающие теги.
                        close_tags: str
                            Закрывающие теги.

                Returns:

                """

        text_list: dict = {}
        text_segm_data: dict = {}
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['max_words']

        for path in sources:
            text_file = self.read_text(file_path=path, lower=True, del_symbols=options['filters'], split=' ',
                                       open_symbol=open_symbol, close_symbol=close_symbol)
            if text_file:
                text_list[path] = self.get_samples(text_file, open_tags, close_tags)

        for key, value in sorted(text_list.items()):
            if options['text_mode'] == LayerTextModeChoice.completely:
                text_segm_data[';'.join([key, f'[0-{options["max_words"]}]'])] = \
                    value[:options['max_words']]
            elif options['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(value)
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text_segm_data[';'.join([key, f'[{cur_step}-{cur_step + length}]'])] = value[
                                                                                           cur_step:cur_step + length]
                    cur_step += options['step']
                    if cur_step + length > max_length:
                        stop_flag = True

        text_sorted = []
        for elem in sorted(text_segm_data.keys()):
            text_sorted.append(text_segm_data[elem])

        instructions = {'instructions': text_sorted,
                        'parameters': {'open_tags': options['open_tags'],
                                       'close_tags': options['close_tags'],
                                       'put': options['put'],
                                       'num_classes': len(open_tags),
                                       'classes_names': open_tags,
                                       'length': length
                                       }
                        }

        return instructions

    def create(self, source: Any, **options):
        if not isinstance(source, list):
            source = literal_eval(source)
        array = []
        if len(source) < options['length']:
            source += [list() for _ in range(options['length'] - len(source))]
        for elem in source:
            tags = [0 for _ in range(options['num_classes'])]
            if elem:
                for cls_name in elem:
                    tags[options['classes_names'].index(cls_name)] = 1
            array.append(tags)
        array = np.array(array, dtype='uint8')

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        return array

    @staticmethod
    def read_text(file_path, lower, del_symbols, split, open_symbol=None, close_symbol=None) -> str:

        text = autodetect_encoding(file_path)

        if open_symbol:
            text = re.sub(open_symbol, f" {open_symbol}", text)
            text = re.sub(close_symbol, f"{close_symbol} ", text)

        text = ' '.join(text_to_word_sequence(text, **{'lower': lower, 'filters': del_symbols, 'split': split}))

        return text

    @staticmethod
    def get_samples(doc_text: str, op_tags, cl_tags):

        indexes = []
        idx = []
        for word in doc_text.split(' '):
            try:
                if word in op_tags:
                    idx.append(op_tags[op_tags.index(word)])
                elif word in cl_tags:
                    idx.remove(op_tags[cl_tags.index(word)])
                else:
                    indexes.append(idx.copy())
            except ValueError:
                pass

        return indexes
