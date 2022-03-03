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
        text_segm: dict = {}
        text_segm_data: dict = {}
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['max_words']

        for path in sources:
            text_file = self.read_text(file_path=path, op_symbol=open_symbol, cl_symbol=close_symbol)
            if text_file:
                text_file = text_file.split(' ')
                text_segm[path] = self.get_samples(text_file, options['filters'], open_tags, close_tags)  # '\r\t\n\ufeff\xa0'
                for word in text_file:
                    if word in open_tags + close_tags:
                        text_file.pop(text_file.index(word))
                for word in text_file:
                    if open_symbol in word or close_symbol in word and word in open_tags + close_tags:
                        text_file.pop(text_file.index(word))
                text_list[path] = text_file

        for key, value in sorted(text_list.items()):
            # value = value.split(' ')
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
                text_segm_data[';'.join([str(key), f'[0-{adjusted_length}]'])] = text_segm[key][0: adjusted_length]
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
                    text_segm_data[';'.join([str(key), f'[{cur_step}-{cur_step + adjusted_length}]'])] =\
                        text_segm[key][cur_step: cur_step + adjusted_length]
                    cur_step += options['step'] + (adjusted_length - length)

        text_sorted = []
        for elem in sorted(text_segm_data.keys()):
            text_sorted.append(text_segm_data[elem])

        instructions = {'instructions': text_sorted,
                        'parameters': {'open_tags': options['open_tags'],
                                       'close_tags': options['close_tags'],
                                       'put': options['put'],
                                       'prepare_method': options['prepare_method'],
                                       'cols_names': options['cols_names'],
                                       'num_classes': len(open_tags),
                                       'classes_names': open_tags,
                                       'length': length
                                       }
                        }

        return instructions

    def create(self, source: Any, **options):

        current_source = source.copy() if isinstance(source, list) else literal_eval(source).copy()
        array = []
        if len(current_source) < options['length']:
            current_source += [list() for _ in range(options['length'] - len(current_source))]
        for elem in current_source:
            if elem is not None:
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
    def read_text(file_path, op_symbol=None, cl_symbol=None) -> str:

        text = autodetect_encoding(file_path)

        if op_symbol:
            text = re.sub(op_symbol, f" {op_symbol}", text)
            text = re.sub(cl_symbol, f"{cl_symbol} ", text)

        text = ' '.join(text_to_word_sequence(text, **{'lower': False, 'filters': '\r\t\n\ufeff\xa0', 'split': ' '}))

        return text

    @staticmethod
    def get_samples(doc_text: list, filters: str, op_tags: list, cl_tags: list):

        segmentation = []
        sample = []
        for elem in doc_text:
            try:
                response = text_to_word_sequence(elem, **{'lower': True, 'filters': filters, 'split': ' '})
                if not response:
                    segmentation.append(None)
                    continue
                if response[0] in op_tags:
                    sample.append(op_tags[op_tags.index(response[0])])
                elif response[0] in cl_tags:
                    sample.remove(op_tags[cl_tags.index(response[0])])
                else:
                    segmentation.append(sample.copy())
            except ValueError:
                pass

        return segmentation
