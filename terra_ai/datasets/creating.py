import os
import random
from typing import Any, Union

import numpy as np
import re
import pymorphy2
import shutil
import json
import joblib

from tqdm.notebook import tqdm
from io import open as io_open
from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone

from terra_ai import out_exchange
from .data import DataType
from . import array_creator, loader
from ..data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList


class CreateDTS(object):

    def __init__(self, trds_path='/content/drive/MyDrive/TerraAI/datasets'):

        self._datatype = 'DIM'

        self.trds_path: str = trds_path
        self.file_folder: str = ''

        self.name: str = ''
        self.source: str = ''
        self.tags: dict = {}
        self.user_tags: list = []
        self.language: str = ''
        self.divide_ratio: list = [0.8, 0.1, 0.1]
        self.limit: int = 0
        self.input_datatype: dict = {}  # string
        self.input_dtype: dict = {}
        self.input_shape: dict = {}
        self.input_names: dict = {}
        self.output_datatype: dict = {}
        self.output_dtype: dict = {}
        self.output_shape: dict = {}
        self.output_names: dict = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.one_hot_encoding: dict = {}
        self.task_type: dict = {}
        self.zip_params: dict = {}
        self.user_parameters: dict = {}
        self.use_generator: bool = False

        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.df: dict = {}
        self.tsgenerator: dict = {}

        self.instructions: dict = {'inputs': {}, 'outputs': {}}
        self.limit: int
        self.dataset: dict = {}

        self.y_cls: np.ndarray = np.array([])
        self.sequence: list = []
        self.peg: list = []
        self.iter: int = 0
        self.mode: str = ''
        self.split_sequence: dict = {}
        self.temporary: dict = {}

        pass

    @property
    def datatype(self) -> str:
        return self._datatype

    @datatype.setter
    def datatype(self, shape: int):
        self._datatype = DataType.get(shape, 'DIM')

    def load_data(self, strict_object):

        loader.load_data(strict_object=strict_object)

        self.zip_params = json.loads(strict_object.json())
        self.file_folder = loader.file_folder

    def set_dataset_data(self, data: Union[CreationInputsList, CreationOutputsList]):
        for elem in data:
            self.tags[elem.alias] = elem.type
            self.input_names[elem.alias] = elem.name
            self.user_parameters[elem.alias] = elem.parameters

    def create_instructions(self, instruction_type: str, data: Union[CreationInputsList, CreationOutputsList]):
        self.iter = 0
        self.mode = instruction_type
        instruction_key = instruction_type + 's'
        for elem in data:
            self.iter += 1
            self.instructions[instruction_key][elem.alias] = getattr(self, f"instructions_{elem.type}")(
                **elem.parameters.dict())

    def create_dataset(self, creation_data: CreationData):
        self.name = creation_data.name
        self.divide_ratio = tuple(creation_data.info.part.dict().values())
        self.source = 'custom dataset'
        self.user_tags = creation_data.tags
        # self.use_generator = creation_data.use_generator

        for data in [creation_data.inputs, creation_data.outputs]:
            self.set_dataset_data(data)

        # Создаем входные инструкции
        self.create_instructions(instruction_type='input', data=creation_data.inputs)
        # Создаем выходные инструкции
        self.create_instructions(instruction_type='output', data=creation_data.outputs)

        # Получаем входные параметры
        for key in self.instructions['inputs'].keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                self.instructions['inputs'][key]['instructions'][0], **self.instructions['inputs'][key]['parameters'])
            self.input_shape[key] = array.shape
            self.input_dtype[key] = str(array.dtype)
            self.datatype = array.shape
            self.input_datatype[key] = self.datatype
        # Получаем выходные параметры
        for key in self.instructions['outputs'].keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                self.instructions['outputs'][key]['instructions'][0], **self.instructions['outputs'][key]['parameters'])
            if isinstance(array, tuple):
                for i in range(len(array)):
                    self.output_shape[key.replace(key[-1], str(int(key[-1])+i))] = array[i].shape
                    self.output_dtype[key.replace(key[-1], str(int(key[-1])+i))] = str(array[i].dtype)
                    self.datatype = array[i].shape
                    self.output_datatype[key.replace(key[-1], str(int(key[-1])+i))] = self.datatype
            else:
                self.output_shape[key] = array.shape
                self.output_dtype[key] = str(array.dtype)
                self.datatype = array.shape
                self.output_datatype[key] = self.datatype

        # Разделение на три выборки
        self.split_sequence['train'] = []
        self.split_sequence['val'] = []
        self.split_sequence['test'] = []
        for i in range(len(self.peg) - 1):
            indices = np.arange(self.peg[i], self.peg[i + 1])
            train_len = int(self.divide_ratio[0] * len(indices))
            val_len = int(self.divide_ratio[1] * len(indices))
            indices = indices.tolist()
            self.split_sequence['train'].extend(indices[:train_len])
            self.split_sequence['val'].extend(indices[train_len:train_len + val_len])
            self.split_sequence['test'].extend(indices[train_len + val_len:])
        if not dataset_dict['parameters']['preserve_sequence']:
            random.shuffle(self.split_sequence['train'])
            random.shuffle(self.split_sequence['val'])
            random.shuffle(self.split_sequence['test'])

        self.limit: int = len(self.instructions['inputs']['input_1']['instructions'])

        data = {}
        if dataset_dict['parameters']['use_generator']:
            # Сохранение датасета для генератора
            data['zip_params'] = self.zip_params
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions'), exist_ok=True)
            for key in self.instructions.keys():
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions', key), exist_ok=True)
                for inp in self.instructions[key].keys():
                    with open(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions', key, f'{inp}.json'),
                              'w') as instruction:
                        json.dump(self.instructions[key][inp], instruction)
            with open(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions', 'sequence.json'),
                      'w') as seq:
                json.dump(self.split_sequence, seq)
            if 'txt_list' in array_creator.__dict__.keys():
                with open(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions', 'txt_list.json'),
                          'w') as fp:
                    json.dump(array_creator.txt_list, fp)
        else:
            # Сохранение датасета с NumPy
            for key in self.instructions['inputs'].keys():
                x: list = []
                for i in range(self.limit):
                    x.append(getattr(array_creator, f"create_{self.tags[key]}")(
                        self.instructions['inputs'][key]['instructions'][i],
                        **self.instructions['inputs'][key]['parameters']))
                self.X['train'][key] = np.array(x)[self.split_sequence['train']]
                self.X['val'][key] = np.array(x)[self.split_sequence['val']]
                self.X['test'][key] = np.array(x)[self.split_sequence['test']]

            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    y_1: list = []
                    y_2: list = []
                    y_3: list = []
                    for i in range(self.limit):
                        arrays = getattr(array_creator, f"create_{self.tags[key]}")(
                                         self.instructions['outputs'][key]['instructions'][i],
                                         **self.instructions['outputs'][key]['parameters'])
                        y_1.append(arrays[0])
                        y_2.append(arrays[1])
                        y_3.append(arrays[2])

                    splits = ['train', 'val', 'test']
                    for spl_seq in splits:
                        for i in range(len(splits)):
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1])))] = np.array(y_1)[self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1])+1))] = np.array(y_2)[self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1])+2))] = np.array(y_3)[self.split_sequence[spl_seq]]
                else:
                    y: list = []
                    for i in range(self.limit):
                        y.append(getattr(array_creator, f"create_{self.tags[key]}")(
                            self.instructions['outputs'][key]['instructions'][i],
                            **self.instructions['outputs'][key]['parameters']))
                    self.Y['train'][key] = np.array(y)[self.split_sequence['train']]
                    self.Y['val'][key] = np.array(y)[self.split_sequence['val']]
                    self.Y['test'][key] = np.array(y)[self.split_sequence['test']]

            for sample in self.X.keys():
                os.makedirs(os.path.join(self.trds_path, 'arrays', sample), exist_ok=True)
                for inp in self.X[sample].keys():
                    os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample), exist_ok=True)
                    joblib.dump(self.X[sample][inp],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample, f'{inp}.gz'))

            for sample in self.Y.keys():
                for inp in self.Y[sample].keys():
                    os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample), exist_ok=True)
                    joblib.dump(self.Y[sample][inp],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample, f'{inp}.gz'))

        if array_creator.scaler:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'scalers'), exist_ok=True)
        if array_creator.tokenizer:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer'), exist_ok=True)
        if array_creator.word2vec:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec'), exist_ok=True)
        # if self.createarray.tsgenerator:
        #     os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tsgenerator'), exist_ok=True)

        for scaler in array_creator.scaler.keys():
            if array_creator.scaler[scaler]:
                joblib.dump(array_creator.scaler[scaler],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'scalers', f'{scaler}.gz'))
        for tok in array_creator.tokenizer.keys():
            if array_creator.tokenizer[tok]:
                joblib.dump(array_creator.tokenizer[tok],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer', f'{tok}.gz'))
        for w2v in array_creator.word2vec.keys():
            if array_creator.word2vec[w2v]:
                joblib.dump(array_creator.word2vec[w2v],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec', f'{w2v}.gz'))
        # for tsg in self.createarray.tsgenerator.keys():
        #     if self.createarray.tsgenerator[tsg]:
        #         joblib.dump(self.createarray.tsgenerator[tsg],
        #                     os.path.join(self.trds_path, f'dataset {self.name}', 'tsgenerator', f'{tsg}.gz'))

        attributes = ['name', 'source', 'tags', 'user_tags', 'language',
                      'input_datatype', 'input_dtype', 'input_shape', 'input_names',
                      'output_datatype', 'output_dtype', 'output_shape', 'output_names',
                      'num_classes', 'classes_names', 'classes_colors',
                      'one_hot_encoding', 'task_type', 'limit', 'use_generator']

        for attr in attributes:
            data[attr] = self.__dict__[attr]
        data['date'] = datetime.now().astimezone(timezone('Europe/Moscow')).isoformat()
        with open(os.path.join(self.trds_path, f'dataset {self.name}', 'config.json'), 'w') as fp:
            json.dump(data, fp)

        pass

    def instructions_images(self, **options):

        instructions: dict = {}
        instr: list = []
        y_cls: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)
        if 'object_detection' in self.tags.values():
            for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['folder_name']))):
                if 'txt' not in file_name:
                    instr.append(os.path.join(options['folder_name'], file_name))
                    peg_idx += 1
                    y_cls.append(cls_idx)
        else:
            path = self.file_folder
            if options['folder_name']:
                path = os.path.join(self.file_folder, options['folder_name'])
            for directory, folder, file_name in sorted(os.walk(path)):
                if file_name:
                    file_folder = directory.replace(self.file_folder, '')[1:]
                    for name in sorted(file_name):
                        instr.append(os.path.join(file_folder, name))
                        peg_idx += 1
                        y_cls.append(cls_idx)
                    cls_idx += 1
                    self.peg.append(peg_idx)
        instructions['instructions'] = instr
        instructions['parameters'] = options
        self.y_cls = y_cls

        return instructions

    def instructions_video(self, **options):

        instructions: dict = {}
        instr: list = []
        y_cls: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)

        path = self.file_folder
        if options['folder_name']:
            path = os.path.join(self.file_folder, options['folder_name'])
        for directory, folder, file_name in sorted(os.walk(path)):
            if file_name:
                file_folder = directory.replace(self.file_folder, '')[1:]
                for name in sorted(file_name):
                    instr.append(os.path.join(file_folder, name))
                    peg_idx += 1
                    if options['class_mode'] == 'По каждому кадру':
                        y_cls.append(np.full((options['max_frames'], 1), cls_idx).tolist())
                    else:
                        y_cls.append(cls_idx)
                cls_idx += 1
                self.peg.append(peg_idx)
        instructions['instructions'] = instr
        instructions['parameters'] = options
        self.y_cls = y_cls

        return instructions

    def instructions_text(self, **options):

        def read_text(file_path):

            del_symbols = ['\n', '\t', '\ufeff']
            if options['delete_symbols']:
                del_symbols += options['delete_symbols'].split(' ')

            with io_open(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
                for del_symbol in del_symbols:
                    text = text.replace(del_symbol, ' ')
            for put, tag in self.tags.items():
                if tag == 'text_segmentation':
                    open_symbol = self.user_parameters[put]['open_tags'].split(' ')[0][0]
                    close_symbol = self.user_parameters[put]['open_tags'].split(' ')[0][-1]
                    text = re.sub(open_symbol, f" {open_symbol}", text)
                    text = re.sub(close_symbol, f"{close_symbol} ", text)
                    break

            return text

        def apply_pymorphy(text, morphy) -> list:

            words_list = text.split(' ')
            words_list = [morphy.parse(w)[0].normal_form for w in words_list]

            return words_list

        txt_list: dict = {}

        if options['folder_name']:
            for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['folder_name']))):
                txt_list[os.path.join(options['folder_name'], file_name)] = read_text(
                    os.path.join(self.file_folder, options['folder_name'], file_name))
        else:
            tree = os.walk(self.file_folder)
            for directory, folder, file_name in sorted(tree):
                if bool(file_name) is not False:
                    folder_name = directory.split(os.path.sep)[-1]
                    for name in sorted(file_name):
                        text_file = read_text(os.path.join(directory, name))
                        if text_file:
                            txt_list[os.path.join(folder_name, name)] = text_file
                else:
                    continue

        #################################################
        if options['pymorphy']:
            pymorphy = pymorphy2.MorphAnalyzer()
            for i in range(len(txt_list)):
                txt_list[i] = apply_pymorphy(txt_list[i], pymorphy)
        #################################################

        filters = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
        for key, value in self.tags.items():
            if value == 'text_segmentation':
                open_tags = self.user_parameters[key]['open_tags']
                close_tags = self.user_parameters[key]['close_tags']
                tags = f'{open_tags} {close_tags}'
                for ch in filters:
                    if ch in set(tags):
                        filters = filters.replace(ch, '')
                break

        array_creator.create_tokenizer(self.mode, self.iter, **{'num_words': options['max_words_count'],
                                                                   'filters': filters,
                                                                   'lower': True,
                                                                   'split': ' ',
                                                                   'char_level': False,
                                                                   'oov_token': '<UNK>'})
        array_creator.tokenizer[f'{self.mode}_{self.iter}'].fit_on_texts(list(txt_list.values()))

        array_creator.txt_list[f'{self.mode}_{self.iter}'] = {}
        for key, value in txt_list.items():
            array_creator.txt_list[f'{self.mode}_{self.iter}'][key] =\
                array_creator.tokenizer[f'{self.mode}_{self.iter}'].texts_to_sequences([value])[0]

        if options['word_to_vec']:
            reverse_tok = {}
            for key, value in array_creator.tokenizer[f'{self.mode}_{self.iter}'].word_index.items():
                reverse_tok[value] = key
            words = []
            for key in array_creator.txt_list[f'{self.mode}_{self.iter}'].keys():
                for lst in array_creator.txt_list[f'{self.mode}_{self.iter}'][key]:
                    tmp = []
                    for word in lst:
                        tmp.append(reverse_tok[word])
                    words.append(tmp)
            array_creator.create_word2vec(mode=self.mode, iteration=self.iter, words=words,
                                             size=options['word_to_vec_size'], window=10, min_count=1, workers=10,
                                             iter=10)

        instr = []
        if 'text_segmentation' not in self.tags.values():
            y_cls = []
            cls_idx = 0
            length = options['x_len']
            stride = options['step']
            peg_idx = 0
            self.peg.append(0)
            for key in sorted(array_creator.txt_list[f'{self.mode}_{self.iter}'].keys()):
                index = 0
                while index + length <= len(array_creator.txt_list[f'{self.mode}_{self.iter}'][key]):
                    instr.append({'file': key, 'slice': [index, index + length]})
                    peg_idx += 1
                    index += stride
                    y_cls.append(cls_idx)
                self.peg.append(peg_idx)
                cls_idx += 1
            self.y_cls = y_cls
        instructions = {'instructions': instr,
                        'parameters': {'bag_of_words': options['bag_of_words'],
                                       'word_to_vec': options['word_to_vec'],
                                       'put': f'{self.mode}_{self.iter}'
                                       }
                        }

        return instructions

    def instructions_audio(self):

        pass

    def instructions_dataframe(self):

        pass

    def instructions_classification(self, **options):

        self.task_type[f'{self.mode}_{self.iter}'] = 'classification'
        self.one_hot_encoding[f'{self.mode}_{self.iter}'] = options['one_hot_encoding']
        for key, value in self.tags.items():
            if value in ['images', 'text', 'audio', 'video']:
                self.classes_names[f'{self.mode}_{self.iter}'] = sorted(os.listdir(self.file_folder)) if not \
                    self.user_parameters[key]['folder_name'] else list(self.user_parameters[key]['folder_name'].split())
                self.num_classes[f'{self.mode}_{self.iter}'] = len(self.classes_names[f'{self.mode}_{self.iter}'])
            elif value in ['dataframe']:
                self.classes_names[f'{self.mode}_{self.iter}'] = self.temporary['classes_names']
                self.num_classes[f'{self.mode}_{self.iter}'] = self.temporary['num_classes']

        instructions: dict = {'parameters': {'num_classes': len(np.unique(self.y_cls)),
                                             'one_hot_encoding': options['one_hot_encoding']},
                              'instructions': self.y_cls}

        return instructions

    def instructions_regression(self):

        pass

    def instructions_segmentation(self, **options):

        instr: list = []

        self.classes_names[f'{self.mode}_{self.iter}'] = options['classes_names']
        self.classes_colors[f'{self.mode}_{self.iter}'] = options['classes_colors']
        self.num_classes[f'{self.mode}_{self.iter}'] = len(options['classes_names'])
        self.one_hot_encoding[f'{self.mode}_{self.iter}'] = True
        self.task_type[f'{self.mode}_{self.iter}'] = 'segmentation'

        for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['folder_name']))):
            instr.append(os.path.join(options['folder_name'], file_name))

        instructions = {'instructions': instr,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': len(options['classes_names']),
                                       'shape': (self.user_parameters['input_1']['height'],
                                                 self.user_parameters['input_1']['width']),
                                       'classes_colors': options['classes_colors']
                                       }
                        }

        return instructions

    def instructions_object_detection(self, **options):

        data = {}
        instructions = {}
        parameters = {}
        class_names = []

        # obj.data
        with open(os.path.join(self.file_folder, 'obj.data'), 'r') as dt:
            d = dt.read()
        for elem in d.split('\n'):
            if elem:
                elem = elem.split(' = ')
                data[elem[0]] = elem[1]

        for key, value in self.tags.items():
            if value == 'images':
                parameters['height'] = self.user_parameters[key]['height']
                parameters['width'] = self.user_parameters[key]['width']
                parameters['num_classes'] = int(data['classes'])

        # obj.names
        with open(os.path.join(self.file_folder, data["names"].split("/")[-1]), 'r') as dt:
            names = dt.read()
        for elem in names.split('\n'):
            if elem:
                class_names.append(elem)

        for i in range(3):
            self.classes_names[f'{self.mode}_{self.iter+i}'] = class_names
            self.num_classes[f'{self.mode}_{self.iter+i}'] = int(data['classes'])

        # list of txt
        txt_list = []
        with open(os.path.join(self.file_folder, data["train"].split("/")[-1]), 'r') as dt:
            images = dt.read()
        for elem in sorted(images.split('\n')):
            if elem:
                idx = elem.rfind('.')
                elem = elem.replace(elem[idx:], '.txt')
                txt_list.append(os.path.join(*elem.split('/')[1:]))
        instructions['instructions'] = txt_list
        instructions['parameters'] = parameters

        return instructions

    def instructions_text_segmentation(self, **options):

        """

        Args:
            **options:
                open_tags: str
                    Открывающие теги.
                close_tags: str
                    Закрывающие теги.

        Returns:

        """

        def get_ohe_samples(list_of_txt, tags_index):

            segment_array = []
            new_list_of_txt = []
            tag_place = [0 for _ in range(len(open_tags))]
            for ex in list_of_txt:
                if ex in tags_index:
                    place = np.argwhere(tags_index == ex)
                    if len(place) != 0:
                        if place[0][0] < len(open_tags):
                            tag_place[place[0][0]] = 1
                        else:
                            tag_place[place[0][0] - len(open_tags)] = 0
                else:
                    new_list_of_txt.append(ex)
                    segment_array.append(np.where(np.array(tag_place) == 1)[0].tolist())

            return new_list_of_txt, segment_array

        instr: list = []
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        tags: list = open_tags + close_tags
        array_creator.txt_list[f'{self.mode}_{self.iter}'] = {}

        for key, value in self.tags.items():
            if value == 'text':
                tags_indexes = np.array([array_creator.tokenizer[key].word_index[idx] for idx in tags])
                for txt_file in array_creator.txt_list[key].keys():
                    text_instr, segment_instr = get_ohe_samples(array_creator.txt_list[key][txt_file], tags_indexes)
                    array_creator.txt_list[f'{self.mode}_{self.iter}'][txt_file] = segment_instr
                    array_creator.txt_list[key][txt_file] = text_instr

                length = self.user_parameters[key]['x_len']
                stride = self.user_parameters[key]['step']
                peg_idx = 0
                self.peg = []
                self.peg.append(0)
                for path in sorted(array_creator.txt_list[f'{self.mode}_{self.iter}'].keys()):
                    index = 0
                    while index + length <= len(array_creator.txt_list[key][path]):
                        instr.append({'file': path, 'slice': [index, index + length]})
                        peg_idx += 1
                        index += stride
                    self.peg.append(peg_idx)
                self.instructions['inputs'][key]['instructions'] = instr

        instructions = {'instructions': instr,
                        'parameters': {'num_classes': len(open_tags),
                                       'put': f'{self.mode}_{self.iter}'}
                        }

        return instructions
