import os
import random
from typing import Any, Union

import numpy as np
import pandas as pd
import re
import pymorphy2
import shutil
import json
import joblib
from pydantic import DirectoryPath
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from librosa import load as librosa_load
import imgaug.augmenters as iaa

from tqdm.notebook import tqdm
from io import open as io_open
from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone
import cv2

# from terra_ai import out_exchange
from ..utils import decamelize
from .data import DataType, Preprocesses, PathsData, InstructionsData, DatasetInstructionsData
from . import array_creator
from . import loading as dataset_loading
from ..data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList, CreationInputData, \
    CreationOutputData
from ..data.datasets.dataset import DatasetData, DatasetLayerData, DatasetInputsData, DatasetOutputsData


class CreateDTS(object):

    def __init__(self):

        self.dataset_user_data: CreationData
        self.paths: PathsData
        self.instructions: InstructionsData
        self.inputs = None
        self.outputs = None
        self.input_names: dict = {}
        self.output_names: dict = {}
        self.trds_path: str = ''
        self.name: str = ''
        self.source: str = ''
        self.tags: dict = {}
        self.user_tags: list = []
        self.limit: int = 0
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.encoding: dict = {}
        self.task_type: dict = {}
        self.user_parameters: dict = {}
        self.sequence: list = []
        self.peg: list = []
        self.split_sequence: dict = {}
        self.use_generator: bool = False

        self.file_folder: str = ''
        self.language: str = ''
        self.y_cls: list = []
        self.mode: str = ''
        self.iter: int = 0

        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.df: dict = {}
        self.tsgenerator: dict = {}
        self.temporary: dict = {}

    def create_dataset(self, creation_data: CreationData):

        self.dataset_user_data = creation_data

        self.name = creation_data.name
        self.user_tags = creation_data.tags
        self.use_generator = creation_data.use_generator
        self.trds_path = creation_data.datasets_path
        self.file_folder = str(creation_data.source_path)
        self.source = 'custom dataset'

        for inp in creation_data.inputs:
            self.set_dataset_data(inp)
        for out in creation_data.outputs:
            self.set_dataset_data(out)

        # Устанавливаем пути
        self.paths = self.set_paths(data=creation_data)

        # Создаем инструкции
        self.instructions = self.create_instructions(creation_data)

        self.limit: int = len(self.instructions.inputs.get(1).instructions)

        # Получаем входные параметры
        self.inputs = self.create_inputs_parameters(creation_data=creation_data)

        # Получаем выходные параметры
        self.outputs = self.create_output_parameters(creation_data=creation_data)

        # Разделение на три выборки
        self.sequence_split(creation_data=creation_data)

        if creation_data.use_generator:
            # Сохранение датасета для генератора
            with open(os.path.join(self.paths.instructions, f'generator_instructions.json'),
                      'w') as instruction:
                json.dump(self.instructions.native(), instruction)
            with open(os.path.join(self.paths.instructions, 'sequence.json'), 'w') as seq:
                json.dump(self.split_sequence, seq)
            if 'text' in self.tags.keys():  # if 'txt_list' in self.createarray.__dict__.keys():
                with open(os.path.join(self.paths.instructions, 'txt_list.json'), 'w') as fp:
                    json.dump(array_creator.txt_list, fp)
        else:
            # Сохранение датасета с NumPy
            x_array = self.create_dataset_arrays(creation_data=creation_data, put_data=self.instructions.inputs)
            y_array = self.create_dataset_arrays(creation_data=creation_data, put_data=self.instructions.outputs)

            self.write_arrays(x_array, y_array)

        # запись препроцессов (скейлер, токенайзер и т.п.)
        self.write_preprocesses_to_files()

        # создание и запись конфигурации датасета
        output = DatasetData(**self.create_dataset_configure(creation_data=creation_data))

        # print(output.json(indent=2))
        return output

    def set_dataset_data(self, layer: Union[CreationInputData, CreationOutputData]):
        self.tags[layer.id] = decamelize(layer.type)
        if isinstance(layer, CreationInputData):
            self.input_names[layer.id] = layer.name
        else:
            self.output_names[layer.id] = layer.name
        self.user_parameters[layer.id] = layer.parameters

    def set_paths(self, data: CreationData) -> PathsData:
        dataset_path = os.path.join(data.datasets_path, f'dataset {data.name}')
        instructions_path = None
        arrays_path = os.path.join(dataset_path, "arrays")
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(arrays_path, exist_ok=True)
        if data.use_generator:
            instructions_path = os.path.join(dataset_path, "instructions")
            os.makedirs(instructions_path, exist_ok=True)
        return PathsData(datasets=dataset_path, instructions=instructions_path, arrays=arrays_path)

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:
        inputs = self.create_put_instructions(data=creation_data.inputs)
        outputs = self.create_put_instructions(data=creation_data.outputs)
        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)
        return instructions

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:
        instructions = {}
        for elem in data:
            instructions_data = InstructionsData(**getattr(self, f"instructions_{decamelize(elem.type)}")(elem))
            instructions.update([(elem.id, instructions_data)])
        return instructions

    def write_preprocesses_to_files(self):
        for preprocess_name in Preprocesses:
            preprocess = getattr(array_creator, preprocess_name)
            preprocess_file_path = os.path.join(self.paths.datasets, preprocess_name)
            if preprocess:
                os.makedirs(preprocess_file_path, exist_ok=True)
                for key in preprocess.keys():
                    if preprocess.get(key, {}):
                        joblib.dump(preprocess[key], os.path.join(preprocess_file_path, f'{key}.gz'))

    def create_inputs_parameters(self, creation_data: CreationData) -> dict:
        creating_inputs_data = {}
        for key in self.instructions.inputs.keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                creation_data.source_path,
                self.instructions.inputs.get(key).instructions[0],
                **self.instructions.inputs.get(key).parameters
            )
            if isinstance(array, tuple):
                for i in range(len(array)):
                    current_input = DatasetInputsData(datatype=DataType.get(len(array[i].shape), 'DIM'),
                                                      dtype=str(array[i].dtype),
                                                      shape=array[i].shape,
                                                      name=self.input_names.get(key, ''),
                                                      task=creation_data.inputs.get(key).type
                                                      )
                    creating_inputs_data.update([(key, current_input.native())])
            else:
                current_input = DatasetInputsData(datatype=DataType.get(len(array.shape), 'DIM'),
                                                  dtype=str(array.dtype),
                                                  shape=array.shape,
                                                  name=self.input_names.get(key, ''),
                                                  task=creation_data.inputs.get(key).type
                                                  )
                creating_inputs_data.update([(key, current_input.native())])
        return creating_inputs_data

    def create_output_parameters(self, creation_data: CreationData) -> dict:
        creating_outputs_data = {}
        for key in self.instructions.outputs.keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                creation_data.source_path,
                self.instructions.outputs.get(key).instructions[0],
                **self.instructions.outputs.get(key).parameters
            )
            if isinstance(array, tuple):
                for i in range(len(array)):
                    current_output = DatasetOutputsData(datatype=DataType.get(len(array[i].shape), 'DIM'),
                                                        dtype=str(array[i].dtype),
                                                        shape=array[i].shape,
                                                        name=self.output_names.get(key, ''),
                                                        task=creation_data.outputs.get(key).type
                                                        )
                    creating_outputs_data.update([(key + i, current_output.native())])
            else:
                current_output = DatasetOutputsData(datatype=DataType.get(len(array.shape), 'DIM'),
                                                    dtype=str(array.dtype),
                                                    shape=array.shape,
                                                    name=self.output_names.get(key, ''),
                                                    task=creation_data.outputs.get(key).type
                                                    )
                creating_outputs_data.update([(key, current_output.native())])
        return creating_outputs_data

    def sequence_split(self, creation_data: CreationData):
        self.split_sequence['train'] = []
        self.split_sequence['val'] = []
        self.split_sequence['test'] = []
        for i in range(len(self.peg) - 1):
            indices = np.arange(self.peg[i], self.peg[i + 1])
            train_len = int(creation_data.info.part.train * len(indices))
            val_len = int(creation_data.info.part.validation * len(indices))
            indices = indices.tolist()
            self.split_sequence['train'].extend(indices[:train_len])
            self.split_sequence['val'].extend(indices[train_len:train_len + val_len])
            self.split_sequence['test'].extend(indices[train_len + val_len:])
        if creation_data.info.shuffle:
            random.shuffle(self.split_sequence['train'])
            random.shuffle(self.split_sequence['val'])
            random.shuffle(self.split_sequence['test'])

    def create_dataset_arrays(self, creation_data: CreationData, put_data: dict) -> dict:
        out_array = {'train': {}, 'val': {}, 'test': {}}
        splits = list(self.split_sequence.keys())
        num_arrays = 1
        for key in put_data.keys():
            current_arrays: list = []
            if self.tags[key] == 'object_detection':
                for i in range(6):
                    globals()[f'current_arrays_{i + 1}'] = []
            for i in range(self.limit):
                array = getattr(array_creator, f"create_{self.tags[key]}")(
                    creation_data.source_path,
                    put_data.get(key).instructions[i],
                    **put_data.get(key).parameters)
                if self.tags[key] == 'object_detection':
                    for j in range(num_arrays):
                        globals()[f'current_arrays_{j + 1}'].append(array[j])
                else:
                    current_arrays.append(array)
            for spl_seq in splits:
                if self.tags[key] == 'object_detection':
                    for i in range(len(splits)):
                        for j in range(num_arrays):
                            out_array[spl_seq][key + i] = np.array(globals()[f'current_arrays_{j + 1}'])[self.split_sequence[spl_seq]]
                else:
                    out_array[spl_seq][key] = np.array(current_arrays)[self.split_sequence[spl_seq]]
        return out_array

    def write_arrays(self, array_x, array_y):
        for array in [array_x, array_y]:
            for sample in array.keys():
                for inp in array[sample].keys():
                    os.makedirs(os.path.join(self.paths.arrays, sample), exist_ok=True)
                    joblib.dump(array[sample][inp], os.path.join(self.paths.arrays, sample, f'{inp}.gz'))

    def create_dataset_configure(self, creation_data: CreationData) -> dict:

        data = {}
        attributes = ['name', 'source', 'tags', 'user_tags', 'language',
                      'inputs', 'outputs', 'num_classes', 'classes_names', 'classes_colors',
                      'encoding', 'task_type', 'use_generator']

        tags = []
        for value in self.tags.values():
            tags.append({'alias': value, 'name': value.title()})
        self.tags = tags

        for attr in attributes:
            data[attr] = self.__dict__[attr]
        data['date'] = datetime.now().astimezone(timezone('Europe/Moscow')).isoformat()
        data['alias'] = creation_data.alias
        with open(os.path.join(self.trds_path, f'dataset {self.name}', 'config.json'), 'w') as fp:
            json.dump(data, fp)
        print(data)

        return data

    def instructions_images_obj_detection(self, folder_name: str) -> list:
        data: list = []
        for file_name in sorted(os.listdir(os.path.join(self.file_folder, folder_name))):
            if 'txt' not in file_name:
                data.append(os.path.join(folder_name, file_name))
        self.y_cls = [0, ]
        return data

    def instructions_images_video(self, folder_name: str, class_mode: bool = False, max_frames: int = None) -> list:
        data: list = []
        y_cls: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)
        path = os.path.join(self.file_folder, folder_name) if folder_name else self.file_folder

        for directory, folder, file_name in sorted(os.walk(path)):
            if file_name:
                file_folder = directory.replace(self.file_folder, '')[1:]
                for name in sorted(file_name):
                    data.append(os.path.join(file_folder, name))
                    peg_idx += 1
                    if class_mode:
                        y_cls.append(np.full((max_frames, 1), cls_idx).tolist())
                    else:
                        y_cls.append(cls_idx)
                cls_idx += 1
                self.peg.append(peg_idx)
        self.y_cls = y_cls
        return data

    def instructions_image(self, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        instr: list = []
        y_cls: list = []
        classes_names: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)
        options['put'] = put_data.id
        if 'object_detection' in self.tags.values():
            options['object_detection'] = True

        if not options['sources_paths'][0].endswith('.csv'):
            for folder_name in options['sources_paths']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            if 'object_detection' in self.tags.values():
                                if 'txt' not in name:
                                    instr.append(os.path.join(file_folder, name))
                                    peg_idx += 1
                            else:
                                instr.append(os.path.join(file_folder, name))
                                peg_idx += 1
                            y_cls.append(cls_idx)
                        cls_idx += 1
                        self.peg.append(peg_idx)
                        classes_names.append(file_folder)
            self.y_cls = y_cls
        elif options['sources_paths'][0].endswith('.csv'):
            for file_name in options['sources_paths']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['cols_name'])
                instr = data[options['cols_name'][0]].to_list()
                prev_elem = instr[0].split('/')[-2]
                for elem in instr:
                    cur_elem = elem.split('/')[-2]
                    if cur_elem != prev_elem:
                        self.peg.append(peg_idx)
                    prev_elem = cur_elem
                    peg_idx += 1
                self.peg.append(len(instr))

        if options.get('augmentation', None):
            aug_parameters = []
            for key, value in options['augmentation'].items():
                aug_parameters.append(getattr(iaa, key)(**value))
            array_creator.augmentation[put_data.id] = iaa.Sequential(aug_parameters, random_order=True)
            del options['augmentation']

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_video(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                put_data: Параметры обработки:
                    height: int
                        Высота кадра.
                    width: int
                        Ширина кадра.
                    fill_mode: int
                        Режим заполнения недостающих кадров (Черными кадрами, Средним значением, Последними кадрами).
                    frame_mode: str
                        Режим обработки кадра (Сохранить пропорции, Растянуть).
            Returns:
                instructions: dict
                    Инструкции для создания массивов.
            """

        options = put_data.parameters.native()
        print(options)
        instructions: dict = {}
        instr: list = []
        paths: list = []
        y_cls: list = []
        classes_names = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)

        if not options['sources_paths'][0].endswith('.csv'):
            for folder_name in options['sources_paths']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            paths.append(os.path.join(file_folder, name))
                        classes_names.append(file_folder)
        elif options['sources_paths'][0].endswith('.csv'):
            for file_name in options['sources_paths']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['cols_name'])
                paths = data[options['cols_name'][0]].to_list()

        prev_class = paths[0].split('/')[-2]
        for idx in range(len(paths)):
            cur_class = paths[idx].split('/')[-2]
            if options['video_mode'] == 'Целиком':
                instr.append({paths[idx]: [0, options['max_frames']]})
                peg_idx += 1
                if cur_class != prev_class:
                    cls_idx += 1
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                y_cls.append(cls_idx)
            elif options['video_mode'] == 'По длине и шагу':
                cur_step = 0
                stop_flag = False
                cap = cv2.VideoCapture(os.path.join(self.file_folder, paths[idx]))
                frame_count = int(cap.get(7))
                while not stop_flag:
                    peg_idx += 1
                    instr.append({paths[idx]: [cur_step, cur_step + options['length']]})
                    if cur_class != prev_class:
                        cls_idx += 1
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    y_cls.append(cls_idx)
                    cur_step += options['step']
                    if cur_step + options['length'] > frame_count:
                        instr.append({paths[idx]: [frame_count - options['length'], frame_count]})
                        stop_flag = True

        self.y_cls = y_cls
        self.peg.append(len(instr))
        for key, value in self.tags.items():
            if value == 'classification':
                self.classes_names[key] = classes_names
                self.num_classes[key] = len(classes_names)

        del options['video_mode']
        del options['length']
        del options['step']
        del options['max_frames']
        options['put'] = put_data.id

        instructions['parameters'] = options
        instructions['instructions'] = instr

        return instructions

    def instructions_text(self, put_data: Union[CreationInputData, CreationOutputData]):

        def read_text(file_path, lower, del_symbols, split, open_symbol=None, close_symbol=None) -> str:

            with open(os.path.join(self.file_folder, file_path), 'r') as txt:
                text = txt.read()

            if open_symbol:
                text = re.sub(open_symbol, f" {open_symbol}", text)
                text = re.sub(close_symbol, f"{close_symbol} ", text)

            text = ' '.join(text_to_word_sequence(text, **{'lower': lower, 'filters': del_symbols, 'split': split}))

            return text

        def apply_pymorphy(text, morphy) -> str:

            words_list = text.split(' ')
            words_list = [morphy.parse(w)[0].normal_form for w in words_list]

            return ' '.join(words_list)

        options = put_data.parameters.native()
        text_mode = options.get('text_mode', '')
        max_words = options.get('max_words', int)
        length = options.get('length', int)
        step = options.get('step', int)
        txt_list: dict = {}
        lower: bool = True
        filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
        split: str = ' '
        open_symbol = None
        close_symbol = None
        tags = None
        classes_names = []

        for i, value in self.tags.items():
            if value == 'text_segmentation':
                open_tags = self.user_parameters.get(i).open_tags
                close_tags = self.user_parameters.get(i).close_tags
                open_symbol = open_tags.split(' ')[0][0]
                close_symbol = open_tags.split(' ')[0][-1]
                tags = f'{open_tags} {close_tags}'
                for ch in filters:
                    if ch in set(tags):
                        filters = filters.replace(ch, '')
                break

        if not options['sources_paths'][0].endswith('.csv'):
            for folder_name in options['sources_paths']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            file_path = os.path.join(file_folder, name)
                            txt_list[file_path] = read_text(file_path, lower, filters, split, open_symbol, close_symbol)
                        classes_names.append(file_folder)
        elif options['sources_paths'][0].endswith('.csv'):
            for file_name in options['sources_paths']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options.get('cols_name', ''))
                column = data[options['cols_name'][0]].to_list()
                for idx, elem in column:
                    txt_list[str(idx)] = elem

        if options.get('pymorphy', ''):
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_list.items():
                txt_list[key] = apply_pymorphy(value, pymorphy)

        if options.get('word_to_vec', bool):
            txt_list_w2v = []
            for elem in list(txt_list.values()):
                txt_list_w2v.append(elem.split(' '))
            array_creator.create_word2vec(put_data.id, txt_list_w2v, **{'size': options.get('word_to_vec_size', ''),
                                                                        'window': 10,
                                                                        'min_count': 1,
                                                                        'workers': 10,
                                                                        'iter': 10})
        else:
            array_creator.create_tokenizer(put_data.id, **{'num_words': options.get('max_words_count', ''),
                                                           'filters': filters,
                                                           'lower': lower,
                                                           'split': split,
                                                           'char_level': False,
                                                           'oov_token': '<UNK>'})
            array_creator.tokenizer[put_data.id].fit_on_texts(list(txt_list.values()))

        instr: list = []
        y_cls: list = []
        peg_idx: int = 0
        cls_idx: int = 0
        prev_class: str = sorted(txt_list.keys())[0].split('/')[-2]
        array_creator.txt_list[put_data.id] = txt_list
        self.peg.append(0)

        for key, value in sorted(txt_list.items()):
            cur_class = key.split('/')[-2]
            if text_mode == 'Целиком':
                instr.append({key: [0, max_words]})
                if cur_class != prev_class:
                    cls_idx += 1
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                peg_idx += 1
                y_cls.append(cls_idx)
            elif text_mode == 'По длине и шагу':
                max_length = len(value.split(' '))
                if 'text_segmentation' in self.tags.values():
                    count = 0
                    for elem in tags.split(' '):
                        count += value.split(' ').count(elem)
                    max_length -= count
                cur_step: int = 0
                stop_flag = False
                while not stop_flag:
                    instr.append({key: [cur_step, cur_step + length]})
                    cur_step += step
                    if cur_class != prev_class:
                        cls_idx += 1
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    peg_idx += 1
                    y_cls.append(cls_idx)
                    if cur_step + options['length'] > max_length:
                        stop_flag = True
        self.peg.append(len(instr))
        self.y_cls = y_cls

        instructions = {'instructions': instr,
                        'parameters': {'embedding': options.get('embedding', bool),
                                       'bag_of_words': options.get('bag_of_words', bool),
                                       'word_to_vec': options.get('word_to_vec', bool),
                                       'put': put_data.id
                                       }
                        }

        return instructions

    def instructions_audio(self, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        sample_rate = options.get('sample_rate', int)
        instructions: dict = {}
        instr: list = []
        paths: list = []
        y_cls: list = []
        classes_names = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)

        if not options['sources_paths'][0].endswith('.csv'):
            for folder_name in options['sources_paths']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            paths.append(os.path.join(file_folder, name))
                        classes_names.append(file_folder)
        elif options['sources_paths'][0].endswith('.csv'):
            for file_name in options['sources_paths']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['cols_name'])
                paths = data[options['cols_name'][0]].to_list()

        prev_class = paths[0].split('/')[-2]
        for idx in range(len(paths)):
            cur_class = paths[idx].split('/')[-2]
            if options.get('audio_mode', '') == 'Целиком':
                instr.append({paths[idx]: [0.0, sample_rate * options.get('max_seconds', int)]})
                if cur_class != prev_class:
                    cls_idx += 1
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                peg_idx += 1
                y_cls.append(cls_idx)
            elif options.get('audio_mode', '') == 'По длине и шагу':
                cur_step = 0.0
                stop_flag = False
                y, sr = librosa_load(path=os.path.join(self.file_folder, paths[idx]), sr=sample_rate, res_type='scipy')
                sample_length = len(y) / sample_rate
                while not stop_flag:
                    instr.append({paths[idx]: [cur_step, cur_step + options.get('length', int)]})
                    if cur_class != prev_class:
                        cls_idx += 1
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    peg_idx += 1
                    y_cls.append(cls_idx)
                    cur_step += options.get('step', int)
                    if cur_step + options.get('length', int) > sample_length:
                        stop_flag = True
        self.peg.append(len(instr))
        self.y_cls = y_cls

        for elem in ['audio_mode', 'file_info', 'length', 'step', 'max_seconds']:
            if elem in options.keys():
                del options[elem]
        instructions['parameters'] = options
        instructions['instructions'] = instr

        return instructions

    def instructions_dataframe(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                **put_data: Параметры датафрейма:
                    MinMaxScaler: строка номеров колонок для обработки
                    StandardScaler: строка номеров колонок для обработки
                    Categorical: строка номеров колонок для обработки c уже готовыми категориями
                    Categorical_ranges: dict для присваивания категории  в зависимости от диапазона данных
                        num_cols: число колонок
                        cols: номера колонок
                        col_(int): строка с диапазонами
                        auto_ranges_(int): строка с номером классов для автоматической категоризации
                    one_hot_encoding: строка номеров колонок для перевода категорий в ОНЕ
                    file_name: имя файла.csv
            Returns:
                instructions: dict      Словарь с инструкциями для create_dataframe.
        """

        def str_to_list(str_numbers, df_cols):
            """
            Получает строку из пользовательских номеров колонок,
            возвращает лист индексов данных колонок
            """
            merged = []
            for i in range(len(str_numbers.split(' '))):
                if '-' in str_numbers.split(' ')[i]:
                    merged.extend(list(range(int(str_numbers.split(' ')[i].split('-')[0]) - 1, int(
                        str_numbers.split(' ')[i].split('-')[1]))))
                elif re.findall(r'\D', str_numbers.split(' ')[i]):
                    merged.append(df_cols.to_list().index(str_numbers.split(' ')[i]))
                else:
                    merged.append(int(str_numbers.split(' ')[i]) - 1)
            return merged

        options = put_data.parameters.native()
        transpose = options['transpose']
        if 'classification' in self.tags.values() and not options['trend']:
            step = 1
            y_col = self.user_parameters[2].cols_names[0]
            if options['pad_sequences'] or options['xlen_step']:
                if options['pad_sequences']:
                    example_length = int(options['example_length'])
                    if transpose:
                        tmp_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             sep=options['separator']).T
                        tmp_df.columns = tmp_df.iloc[0]
                        tmp_df.drop(tmp_df.index[[0]], inplace=True)
                        df_y = tmp_df.loc[:, list(range(example_length+1))]
                    else:
                        df_y = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                           usecols=list(range(example_length+1)),
                                           sep=options['separator'])

                    df_y.fillna(0, inplace=True)
                    df_y.sort_values(by=y_col, inplace=True, ignore_index=True)
                    self.peg.append(0)
                    for i in range(len(df_y) - 1):
                        if df_y[y_col][i] != df_y[y_col][i + 1]:
                            self.peg.append(i + 1)
                    self.peg.append(len(df_y))

                    array_creator.df = df_y.iloc[:, 1:].values
                elif options['xlen_step']:
                    xlen = int(options['xlen'])
                    step_len = int(options['step_len'])
                    if transpose:
                        df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator']).T
                        df.columns = df.iloc[0]
                        df.drop(df.index[[0]], inplace=True)
                    else:
                        df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator'])
                    df.sort_values(by=y_col, inplace=True, ignore_index=True)
                    xlen_array = []
                    for i in range(len(df)):
                        subdf = df.iloc[i, 1:]
                        subdf = subdf.dropna().values.tolist()
                        for j in range(0, len(subdf), step_len):
                            if len(subdf[i:i + step_len]) < xlen:
                                xlen_array.append(subdf[-xlen:])
                                self.y_cls.append(i)
                            else:
                                xlen_array.append(subdf[i:i + xlen])
                                self.y_cls.append(i)
                    array_creator.df = np.array(xlen_array)

                    self.peg.append(0)
                    for i in range(len(self.y_cls) - 1):
                        if self.y_cls[i] != self.y_cls[i + 1]:
                            self.peg.append(i + 1)
                    self.peg.append(len(self.y_cls))

                if 'min_max_scaler' in options.values():
                    array_creator.scaler[put_data.id] = MinMaxScaler()
                    array_creator.scaler[put_data.id].fit(
                        array_creator.df.reshape(-1, 1))

                elif 'standard_scaler' in options.values():
                    array_creator.scaler[put_data.id] = StandardScaler()
                    array_creator.scaler[put_data.id].fit(
                        array_creator.df.reshape(-1, 1))

                instructions = {'parameters': {'scaler': options['scaler'], 'put': put_data.id}}

            else:
                if transpose:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             sep=options['separator']).T
                    general_df.columns = general_df.iloc[0]
                    general_df.drop(general_df.index[[0]], inplace=True)
                    df_with_y = general_df.iloc[:, str_to_list(options['cols_names'][0],
                                                               general_df.columns) + str_to_list(y_col,
                                                                                                 general_df.columns)]
                else:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             nrows=1,
                                             sep=options['separator'])
                    df_with_y = pd.read_csv(
                        os.path.join(self.file_folder, options['sources_paths'][0]), usecols=(str_to_list(
                            options['cols_names'][0], general_df.columns) + str_to_list(y_col,
                                                                                        general_df.columns)),
                        sep=options['separator'])

                df_with_y.sort_values(by=y_col, inplace=True, ignore_index=True)

                self.peg.append(0)
                for i in range(len(df_with_y.loc[:, y_col]) - 1):
                    if df_with_y.loc[:, y_col][i] != df_with_y.loc[:, y_col][i + 1]:
                        self.peg.append(i + 1)
                self.peg.append(len(df_with_y))

                array_creator.df = df_with_y.iloc[:, str_to_list(options['cols_names'][0], df_with_y.columns)]
                instructions = {'parameters': {}}
            stop = len(array_creator.df)
        elif 'timeseries' in self.tags.values() or options['trend']:
            if 'timeseries' in self.tags.values():
                length = int(self.user_parameters[2].length)
                depth = int(self.user_parameters[2].depth)
                step = int(self.user_parameters[2].step)
            elif options['trend']:
                length = int(options['length'])
                depth = 1
                step = int(options['step'])

            if transpose:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator']).T
                general_df.columns = general_df.iloc[0]
                general_df.drop(general_df.index[[0]], inplace=True)
                array_creator.df = general_df.iloc[:, str_to_list(options['cols_names'][0],
                                                                  general_df.columns)]
            else:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         nrows=1,
                                         sep=options['separator'])
                array_creator.df = pd.read_csv(
                    os.path.join(self.file_folder, options['sources_paths'][0]), usecols=(str_to_list(
                        options['cols_names'][0], general_df.columns)),
                    sep=options['separator'])

            stop = len(array_creator.df) - length - depth
            instructions = {'parameters': {'timeseries': True,
                                           'length': length,
                                           'depth': depth,
                                           'step': step}}

            self.peg.append(0)
            self.peg.append(len(np.arange(0, len(array_creator.df) - length - depth - 1, step)))
        else:
            step = 1
            if transpose:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator']).T
                general_df.columns = general_df.iloc[0]
                general_df.drop(general_df.index[[0]], inplace=True)
                array_creator.df = general_df.iloc[:, str_to_list(options['cols_names'][0], general_df.columns)]
            else:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         nrows=1,
                                         sep=options['separator'])
                array_creator.df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                               usecols=(str_to_list(options['cols_names'][0], general_df.columns)),
                                               sep=options['separator'])

            self.peg.append(0)
            self.peg.append(len(array_creator.df))
            instructions = {'parameters': {}}
            stop = len(array_creator.df)
        if options['MinMaxScaler'] or options['StandardScaler']:
            array_creator.scaler[put_data.id] = {'MinMaxScaler': {},
                                                 'StandardScaler': {}}
            if options['MinMaxScaler']:
                instructions['parameters']['MinMaxScaler'] = str_to_list(options['MinMaxScaler'], array_creator.df.columns)
                for i in instructions['parameters']['MinMaxScaler']:
                    array_creator.scaler[put_data.id]['MinMaxScaler'][f'col_{i+1}'] = MinMaxScaler()
                    array_creator.scaler[put_data.id]['MinMaxScaler'][f'col_{i+1}'].fit(
                        array_creator.df.iloc[:, [i]].to_numpy().reshape(-1, 1))

            if options['StandardScaler']:
                instructions['parameters']['StandardScaler'] = str_to_list(options['StandardScaler'], array_creator.df.columns)
                for i in instructions['parameters']['StandardScaler']:
                    array_creator.scaler[put_data.id]['StandardScaler'][f'col_{i+1}'] = StandardScaler()
                    array_creator.scaler[put_data.id]['StandardScaler'][f'col_{i+1}'].fit(
                        array_creator.df.iloc[:, [i]].to_numpy().reshape(-1, 1))

        if options['Categorical']:
            instructions['parameters']['Categorical'] = {}
            instructions['parameters']['Categorical']['lst_cols'] = str_to_list(options['Categorical'],
                                                                                array_creator.df.columns)
            for i in instructions['parameters']['Categorical']['lst_cols']:
                instructions['parameters']['Categorical'][f'col_{i}'] = np.unique(
                    array_creator.df.iloc[:, i]).tolist()

        if options['Categorical_ranges']:
            self.minvalues = {}
            self.maxvalues = {}

            instructions['parameters']['Categorical_ranges'] = {}
            instructions['parameters']['Categorical_ranges']['lst_cols'] = str_to_list(
                options['Categorical_ranges'], array_creator.df.columns)
            for i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                self.minvalues[f'col_{i + 1}'] = array_creator.df.iloc[:, i].min()
                self.maxvalues[f'col_{i + 1}'] = array_creator.df.iloc[:, i].max()
                instructions['parameters']['Categorical_ranges'][f'col_{i}'] = {}
                if len(options['cat_cols'][f'{i + 1}'].split(' ')) == 1:
                    for j in range(int(options['cat_cols'][f'{i + 1}'])):
                        if (j + 1) == int(options['cat_cols'][f'{i + 1}']):
                            instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = \
                                array_creator.df.iloc[:, i].max()
                        else:
                            instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = ((
                                        array_creator.df.iloc[:, i].max() - array_creator.df.iloc[:, i].min()) / int(
                                options['cat_cols'][f'{i + 1}'])) * (j + 1)
                else:
                    for j in range(len(options['cat_cols'][f'{i + 1}'].split(' '))):
                        instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = float(
                            options['cat_cols'][f'{i + 1}'].split(' ')[j])

        if options['one_hot_encoding']:
            instructions['parameters']['one_hot_encoding'] = {}
            instructions['parameters']['one_hot_encoding']['lst_cols'] = str_to_list(options['one_hot_encoding'],
                                                                                     array_creator.df.columns)
            for i in instructions['parameters']['one_hot_encoding']['lst_cols']:
                if options['Categorical_ranges'] and i in str_to_list(
                        options['Categorical_ranges'], array_creator.df.columns):
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        instructions['parameters']['Categorical_ranges'][f'col_{i}'])
                else:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        np.unique(array_creator.df.iloc[:, i]))

        instructions['instructions'] = np.arange(0, stop, step).tolist()
        instructions['parameters']['put'] = put_data.id
        array_creator.df = np.array(array_creator.df)
        return instructions

    def instructions_timeseries(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                **put_data: Параметры временного ряда:
                    length: количество примеров для обучения
                    scaler: скейлер
                    y_cols: колонки для предсказания
                    depth: количество значений для предсказания
                    file_name: имя файла.csv
            Returns:
                instructions: dict      Словарь с инструкциями для create_timeseries.
        """
        options = put_data.parameters.native()
        instructions = {'parameters': {}}
        instructions['parameters']['length'] = int(options['length'])
        instructions['parameters']['scaler'] = options['scaler']
        instructions['parameters']['y_cols'] = options['cols_names'][0]
        instructions['parameters']['depth'] = int(options['depth'])
        step = int(options['step'])

        transpose = self.user_parameters[1].transpose
        if transpose:
            tmp_df_ts = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                    sep=options['separator']).T
            tmp_df_ts.columns = tmp_df_ts.iloc[0]
            tmp_df_ts.drop(tmp_df_ts.index[[0]], inplace=True)
            array_creator.y_subdf = tmp_df_ts.loc[:, instructions['parameters']['y_cols'].split(' ')].values
        else:
            array_creator.y_subdf = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                                sep=options['separator'],
                                                usecols=instructions['parameters']['y_cols'].split(' ')).values

        instructions['parameters']['put'] = put_data.id
        instructions['instructions'] = np.arange(0, (len(array_creator.y_subdf) -
                                                     instructions['parameters']['length'] -
                                                     instructions['parameters']['depth']), step).tolist()

        if 'min_max_scaler' in instructions['parameters'].values():
            array_creator.scaler[put_data.id] = MinMaxScaler()
            array_creator.scaler[put_data.id].fit(
                array_creator.y_subdf.reshape(-1, 1))

        elif 'standard_scaler' in instructions['parameters'].values():
            array_creator.scaler[put_data.id] = StandardScaler()
            array_creator.scaler[put_data.id].fit(
                array_creator.y_subdf.reshape(-1, 1))

        return instructions

    def instructions_classification(self, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        self.task_type[put_data.id] = put_data.type
        self.encoding[put_data.id] = 'ohe' if options['one_hot_encoding'] else None

        if options['sources_paths'][0].endswith('.csv'):
            transpose = self.user_parameters.get(1).transpose
            bool_trend = self.user_parameters.get(1).trend
            if not any(self.y_cls):
                if bool_trend:
                    trend_limit = self.user_parameters.get(1).trend_limit
                    length = self.user_parameters.get(1).length
                    step = self.user_parameters.get(1).step
                    if transpose:
                        tmp_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             sep=options['separator']).T
                        tmp_df.columns = tmp_df.iloc[0]
                        tmp_df.drop(tmp_df.index[[0]], inplace=True)
                        trend_subdf = tmp_df.loc[:, options['cols_names'][0].split(' ')].values
                    else:
                        trend_subdf = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                                  sep=options['separator'],
                                                  usecols=options['cols_names'][0].split(' ')).values

                    if '%' in trend_limit:
                        trend_limit = float(trend_limit[:trend_limit.find('%')])
                        for i in range(0, len(trend_subdf) - length, step):
                            if abs((trend_subdf[i + length + 1] - trend_subdf[i]) /
                                   trend_subdf[i]) * 100 <= trend_limit:
                                self.y_cls.append(0)
                            elif trend_subdf[i + length + 1] > trend_subdf[i]:
                                self.y_cls.append(1)
                            else:
                                self.y_cls.append(2)
                    else:
                        trend_limit = float(trend_limit)
                        for i in range(0, len(trend_subdf) - length, step):
                            if abs(trend_subdf[i + length + 1] - trend_subdf[i]) <= trend_limit:
                                self.y_cls.append(0)
                            elif trend_subdf[i + length + 1] > trend_subdf[i]:
                                self.y_cls.append(1)
                            else:
                                self.y_cls.append(2)

                    if options['one_hot_encoding']:
                        tmp_uniq = list(np.unique(self.y_cls))
                        for i in range(len(self.y_cls)):
                            self.y_cls[i] = tmp_uniq.index(self.y_cls[i])

                    self.classes_names[put_data.id] = np.unique(self.y_cls).tolist()
                    self.num_classes[put_data.id] = len(self.classes_names[put_data.id])
                else:
                    if options['categorical']:
                        for file_name in options['sources_paths']:
                            if transpose:
                                tmp_df = pd.read_csv(os.path.join(self.file_folder, file_name), sep=options['separator']).T
                                tmp_df.columns = tmp_df.iloc[0]
                                tmp_df.drop(tmp_df.index[[0]], inplace=True)
                                data = tmp_df.loc[:, options['cols_names'][0].split(' ')]
                            else:
                                data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                                   usecols=options['cols_names'],
                                                   sep=options['separator'])
                            column = data[options['cols_names'][0]].to_list()
                            classes_names = []
                            for elem in column:
                                if elem not in classes_names:
                                    classes_names.append(elem)
                            self.classes_names[put_data.id] = classes_names
                            self.num_classes[put_data.id] = len(classes_names)
                            for elem in column:
                                self.y_cls.append(classes_names.index(elem))
                    elif options['categorical_ranges']:
                        file_name = options['sources_paths'][0]
                        if transpose:
                            tmp_df = pd.read_csv(os.path.join(self.file_folder, file_name), sep=options['separator']).T
                            tmp_df.columns = tmp_df.iloc[0]
                            tmp_df.drop(tmp_df.index[[0]], inplace=True)
                            data = tmp_df.loc[:, options['cols_names'][0].split(' ')]
                        else:
                            data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                               usecols=options['cols_names'],
                                               sep=options['separator'])
                        column = data[options['cols_names'][0]].to_list()
                        self.minvalue_y = min(column)
                        self.maxvalue_y = max(column)
                        if options['auto_ranges']:
                            border = max(column) / int(options['auto_ranges'])
                            self.classes_names[put_data.id] = np.linspace(
                                border, self.maxvalue_y, int(options['auto_ranges'])).tolist()
                        else:
                            self.classes_names[put_data.id] = options['ranges'].split(' ')

                        self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

                        for elem in column:
                            for i in range(len(self.classes_names[put_data.id])):
                                if elem <= int(self.classes_names[put_data.id][i]):
                                    self.y_cls.append(i)
                                    break
            else:
                if transpose:
                    data = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                       sep=options['separator'],
                                       nrows=1).values
                else:
                    data = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                       sep=options['separator'],
                                       usecols=[0]).values
                self.classes_names[put_data.id] = sorted(np.unique(data).tolist())
                self.num_classes[put_data.id] = len(self.classes_names[put_data.id])
        else:
            if any(i in self.tags.values() for i in ['image', 'text', 'audio', 'video']):
                self.classes_names[put_data.id] = sorted(options['sources_paths'])
                self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

        instructions['parameters'] = {'num_classes': self.num_classes[put_data.id],
                                      'one_hot_encoding': options['one_hot_encoding']}
        instructions['instructions'] = self.y_cls

        return instructions

    def instructions_regression(self, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        instr: list = []

        self.encoding[put_data.id] = None
        self.task_type[put_data.id] = put_data.type

        for file_name in options['sources_paths']:
            data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['cols_names'])
            instr = data[options['cols_names'][0]].to_list()

        if options['scaler'] == 'min_max_scaler' or options['scaler'] == 'standard_scaler':
            if options['scaler'] == 'min_max_scaler':
                array_creator.scaler[put_data.id] = MinMaxScaler()
            if options['scaler'] == 'standard_scaler':
                array_creator.scaler[put_data.id] = StandardScaler()
            array_creator.scaler[put_data.id].fit(np.array(instr).reshape(-1, 1))

        instructions['instructions'] = instr
        instructions['parameters'] = options
        instructions['parameters']['put'] = put_data.id

        return instructions

    def instructions_segmentation(self, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instr: list = []
        shape: tuple = ()

        self.classes_names[put_data.id] = options['classes_names']
        self.classes_colors[put_data.id] = options['classes_colors']
        self.num_classes[put_data.id] = len(options['classes_names'])
        self.encoding[put_data.id] = 'ohe'
        self.task_type[put_data.id] = put_data.type

        for key, value in self.tags.items():
            if value == 'image':
                shape = (self.user_parameters.get(key).height, self.user_parameters.get(key).width)

        if not options['sources_paths'][0].endswith('.csv'):
            for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['sources_paths'][0]))):
                instr.append(os.path.join(options['sources_paths'][0], file_name))
        elif options['sources_paths'][0].endswith('.csv'):
            for file_name in options['sources_paths']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['cols_names'])
                instr = data[options['cols_names'][0]].to_list()

        instructions = {'instructions': instr,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': len(options['classes_names']),
                                       'shape': shape,
                                       'classes_colors': options['classes_colors']
                                       }
                        }

        return instructions

    def instructions_object_detection(self, put_data: Union[CreationInputData, CreationOutputData]):

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

        # for key, value in self.tags.items():
        #     if value == 'images':
        #         parameters['height'] = self.user_parameters.get(key).height
        #         parameters['width'] = self.user_parameters.get(key).width
        parameters['num_classes'] = int(data['classes'])

        # obj.names
        with open(os.path.join(self.file_folder, data["names"].split("/")[-1]), 'r') as dt:
            names = dt.read()
        for elem in names.split('\n'):
            if elem:
                class_names.append(elem)

        for i in range(3):
            self.classes_names[put_data.id] = class_names
            self.num_classes[put_data.id] = int(data['classes'])

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

    def instructions_text_segmentation(self, put_data: Union[CreationInputData, CreationOutputData]):

        """

        Args:
            **put_data:
                open_tags: str
                    Открывающие теги.
                close_tags: str
                    Закрывающие теги.

        Returns:

        """

        def get_samples(doc_text: str, op_tags, cl_tags):

            words = []
            indexes = []
            idx = []
            for word in doc_text.split(' '):
                try:
                    if word in open_tags:
                        idx.append(op_tags.index(word))
                    elif word in close_tags:
                        idx.remove(cl_tags.index(word))
                    else:
                        words.append(word)
                        indexes.append(idx.copy())
                except ValueError:
                    print(word)

            words = ' '.join(words)

            return words, indexes

        options = put_data.parameters.native()
        instr: list = []
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        array_creator.txt_list[put_data.id] = {}
        self.encoding[put_data.id] = 'multi'

        for i, value in self.tags.items():
            if value == 'text':
                for txt_file in array_creator.txt_list.get(i).keys():
                    text_instr, segment_instr = get_samples(array_creator.txt_list.get(i)[txt_file],
                                                            open_tags, close_tags)
                    array_creator.txt_list.get(i)[txt_file] = text_instr
                    array_creator.txt_list[put_data.id][txt_file] = segment_instr

                length = self.user_parameters.get(i).dict()['length']
                step = self.user_parameters.get(i).dict()['step']
                text_mode = self.user_parameters.get(i).dict()['text_mode']
                max_words = self.user_parameters.get(i).dict()['max_words']

                for key, text in sorted(array_creator.txt_list.get(i).items()):
                    if text_mode == 'Целиком':
                        instr.append({key: [0, max_words]})
                    elif text_mode == 'По длине и шагу':
                        max_length = len(text.split(' '))
                        cur_step = 0
                        stop_flag = False
                        while not stop_flag:
                            instr.append({key: [cur_step, cur_step + length]})
                            cur_step += step
                            if cur_step + length > max_length:
                                stop_flag = True

        instructions = {'instructions': instr,
                        'parameters': {'num_classes': len(open_tags),
                                       'put': put_data.id}
                        }

        return instructions
