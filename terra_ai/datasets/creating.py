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
        self.one_hot_encoding: dict = {}
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

    def create_dataset(self, creation_data: CreationData) -> DatasetData:

        self.dataset_user_data = creation_data

        self.name = creation_data.name
        self.user_tags = creation_data.tags
        self.use_generator = creation_data.use_generator
        self.trds_path = creation_data.datasets_path
        self.file_folder = str(creation_data.source_path)

        self.source = 'custom dataset'

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

        data = {}
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

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:
        self.iter = 0
        self.mode = "input" if isinstance(data, CreationInputsList) else "output"
        instructions = {}
        for elem in data:
            self.set_dataset_data(elem)
            self.iter += 1
            instructions_data = InstructionsData(**getattr(self, f"instructions_{decamelize(elem.type)}"
                                                           )(elem))
            instructions.update([(elem.id, instructions_data)])
        return instructions

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:
        inputs = self.create_put_instructions(data=creation_data.inputs)
        outputs = self.create_put_instructions(data=creation_data.outputs)
        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)
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
                    creating_outputs_data.update([(key + i, current_output)])
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
        for key in put_data.keys():
            current_arrays: list = []
            for i in range(self.limit):
                array = getattr(array_creator, f"create_{self.tags[key]}")(
                    creation_data.source_path,
                    put_data.get(key).instructions[i],
                    **put_data.get(key).parameters)
                if self.tags[key] == 'object_detection':
                    for j in range(len(splits)):
                        current_arrays.append(array[j])
                else:
                    current_arrays.append(array)
            for spl_seq in splits:
                if self.tags[key] == 'object_detection':
                    for i in range(len(splits)):
                        out_array[spl_seq][key + i] = np.array(current_arrays[i])[self.split_sequence[spl_seq]]
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
                      'one_hot_encoding', 'task_type', 'limit', 'use_generator']

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

    # def instructions_images(self, **options: Any):
    #     folder = options.get('folder_name', '')
    #     instructions: dict = {}
    #     options['put'] = f'{self.mode}_{self.iter}'
    #
    #     if 'object_detection' in self.tags.values():
    #         instr = self.instructions_images_obj_detection(folder)
    #     else:
    #         instr = self.instructions_images_video(folder)
    #
    #     instructions['instructions'] = instr
    #     instructions['parameters'] = options
    #
    #     return instructions

    def instructions_image(self, put_data: Union[CreationInputData, CreationOutputData]):
        options = put_data.parameters.native()
        instructions: dict = {}
        instr: list = []
        y_cls: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)
        options['put'] = f'{self.mode}_{self.iter}'
        if 'object_detection' in self.tags.values():
            options['object_detection'] = True
        for source_path in options.get('sources_paths', []):
            if source_path.is_dir() or source_path.suffix[1:] != 'csv':
                for directory, folder, file_name in sorted(os.walk(os.path.join(source_path))):
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
                self.y_cls = y_cls
            else:
                for file_name in options['file_info']['path']:
                    data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                       usecols=options['file_info']['cols_name'])
                    instr = data[options['file_info']['cols_name'][0]].to_list()
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
            array_creator.augmentation[f'{self.mode}_{self.iter}'] = iaa.Sequential(aug_parameters,
                                                                                    random_order=True)
            del options['augmentation']

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_video(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                **options: Параметры обработки:
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
        # Сделано Алексеем

        # if options.get('class_mode', '') == 'По каждому кадру':
        #     class_mode = True
        #     max_frames = options.get('max_frames', '')
        #
        # instructions['instructions'] = self.instructions_images_video(folder, class_mode, max_frames)
        options = put_data.parameters.native()
        folder = options.get('folder_name', '')
        class_mode = False
        max_frames = options.get('max_frames', '')
        instructions: dict = {}
        instr: list = []
        paths: list = []
        y_cls: list = []
        classes_names = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)

        options['put'] = f'{self.mode}_{self.iter}'
        if options['file_info']['path_type'] == 'path_folder':
            for folder_name in options['file_info']['path']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            paths.append(os.path.join(file_folder, name))
                        classes_names.append(file_folder)
            self.y_cls = y_cls
        elif options['file_info']['path_type'] == 'path_file':
            for file_name in options['file_info']['path']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['file_info']['cols_name'])
                paths = data[options['file_info']['cols_name'][0]].to_list()

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
                        stop_flag = True

        self.peg.append(len(instr))

        del options['video_mode']
        del options['file_info']
        del options['length']
        del options['step']
        del options['max_frames']
        instructions['parameters'] = options
        instructions['instructions'] = instr

        return instructions

    def instructions_text(self, **options):

        folder_name = options.get('folder_name', '')
        word_to_vec = options.get('word_to_vec', '')
        bag_of_words = options.get('bag_of_words', '')

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

        if folder_name:
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
            array_creator.txt_list[f'{self.mode}_{self.iter}'][key] = \
                array_creator.tokenizer[f'{self.mode}_{self.iter}'].texts_to_sequences([value])[0]

        if word_to_vec:
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
                        'parameters': {'bag_of_words': bag_of_words,
                                       'word_to_vec': word_to_vec,
                                       'put': f'{self.mode}_{self.iter}'
                                       }
                        }

        return instructions

    def instructions_audio(self):

        pass

    def instructions_dataframe(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                **options: Параметры датафрейма:
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

        transpose = options['file_info']['transpose']
        for key, value in self.tags.items():
            if value == 'classification' and not options['trend']:
                step = 1
                y_col = self.user_parameters[key]['file_info']['cols_name'][0]
                if 'pad_sequences' in options.keys() or 'xlen_step' in options.keys():
                    if 'pad_sequences' in options.keys():
                        example_lengh = int(options['example_lengh'])
                        if transpose:
                            tmp_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                 sep=options['file_info']['separator']).T
                            tmp_df.columns = tmp_df.iloc[0]
                            tmp_df.drop(tmp_df.index[[0]], inplace=True)
                            df_y = tmp_df.loc[:, list(range(example_lengh))]
                        else:
                            df_y = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                usecols=list(range(example_lengh)),
                                                sep=options['file_info']['separator'])

                        df_y.fillna(0, inplace=True)
                        df_y.sort_values(by=y_col, inplace=True, ignore_index=True)
                        self.peg.append(0)
                        for i in range(len(df_y) - 1):
                            if df_y[y_col][i] != df_y[y_col][i + 1]:
                                self.peg.append(i + 1)
                        self.peg.append(len(df_y))

                        array_creator.df = df_y.iloc[:, 1:].values
                    elif 'xlen_step' in options.keys():
                        xlen = int(options['xlen'])
                        step_len = int(options['step_len'])
                        if transpose:
                            df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                 sep=options['file_info']['separator']).T
                            df.columns = df.iloc[0]
                            df.drop(df.index[[0]], inplace=True)
                        else:
                            df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                sep=options['file_info']['separator'])
                        xlen_array = []
                        for i in range(len(df)):
                            subdf = df.iloc[i, 1:]
                            subdf = subdf.dropna().values.tolist()
                            for j in range(0, len(subdf), step_len):
                                if len(subdf[i:i+step_len]) < xlen:
                                    xlen_array.append(subdf[-xlen:])
                                    self.y_cls.append(i)
                                else:
                                    xlen_array.append(subdf[i:i+xlen])
                                    self.y_cls.append(i)
                        self.y_cls = np.array(self.y_cls)
                        array_creator.df = np.array(xlen_array)

                        self.peg.append(0)
                        for i in range(len(self.y_cls) - 1):
                            if self.y_cls[i] != self.y_cls[i + 1]:
                                self.peg.append(i + 1)
                        self.peg.append(len(self.y_cls))

                    if 'MinMaxScaler' in options.values():
                        array_creator.scaler[f'{self.mode}_{self.iter}'] = MinMaxScaler()
                        array_creator.scaler[f'{self.mode}_{self.iter}'].fit(
                            array_creator.df.reshape(-1, 1))

                    elif 'StandardScaler' in options.values():
                        array_creator.scaler[f'{self.mode}_{self.iter}'] = StandardScaler()
                        array_creator.scaler[f'{self.mode}_{self.iter}'].fit(
                            array_creator.df.reshape(-1, 1))

                    instructions = {'parameters': {'scaler': options['scaler'], 'put': f'{self.mode}_{self.iter}'}}

                else:
                    if transpose:
                        general_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                 sep=options['file_info']['separator']).T
                        general_df.columns = general_df.iloc[0]
                        general_df.drop(general_df.index[[0]], inplace=True)
                        df_with_y = general_df.iloc[:, str_to_list(options['file_info']['cols_name'][0],
                                                                          general_df.columns) + str_to_list(y_col,
                                                                                                general_df.columns)]
                    else:
                        general_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                    nrows=1,
                                                    sep=options['file_info']['separator'])
                        df_with_y = pd.read_csv(
                            os.path.join(self.file_folder, options['file_info']['path'][0]), usecols=(str_to_list(
                                options['file_info']['cols_name'][0], general_df.columns) + str_to_list(y_col,
                                                                                                general_df.columns)),
                            sep=options['file_info']['separator'])

                    df_with_y.sort_values(by=y_col, inplace=True, ignore_index=True)

                    self.peg.append(0)
                    for i in range(len(df_with_y.loc[:, y_col]) - 1):
                        if df_with_y.loc[:, y_col][i] != df_with_y.loc[:, y_col][i + 1]:
                            self.peg.append(i + 1)
                    self.peg.append(len(df_with_y))

                    array_creator.df = df_with_y.iloc[:, str_to_list(
                        options['file_info']['cols_name'][0], df_with_y.columns)]

                    instructions = {'parameters': {}}
                stop = len(array_creator.df)

            elif value == 'timeseries' or options['trend']:
                try:
                    lengh = int(self.user_parameters[key]['lengh'])
                    depth = int(self.user_parameters[key]['depth'])
                    step = int(self.user_parameters[key]['step'])
                except:
                    lengh = int(options['lengh'])
                    depth = 1
                    step = int(options['step'])

                if transpose:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                    sep=options['file_info']['separator']).T
                    general_df.columns = general_df.iloc[0]
                    general_df.drop(general_df.index[[0]], inplace=True)
                    array_creator.df = general_df.iloc[:, str_to_list(options['file_info']['cols_name'][0],
                                       general_df.columns)]
                else:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                nrows=1,
                                                sep=options['file_info']['separator'])
                    array_creator.df = pd.read_csv(
                            os.path.join(self.file_folder, options['file_info']['path'][0]), usecols=(str_to_list(
                                options['file_info']['cols_name'][0], general_df.columns)),
                            sep=options['file_info']['separator'])

                stop = len(array_creator.df) - lengh - depth
                instructions = {'parameters': {'timeseries': True,
                                                'lengh': lengh,
                                                'depth': depth,
                                                'step': step}}

                self.peg.append(0)
                self.peg.append(len(np.arange(0, len(array_creator.df) - lengh - depth - 1, step)))

        if 'MinMaxScaler' in options.keys() or 'StandardScaler' in options.keys():
            array_creator.scaler[f'{self.mode}_{self.iter}'] = {}
            if 'MinMaxScaler' in options.keys():
                instructions['parameters']['MinMaxScaler'] = str_to_list(options['MinMaxScaler'],
                                                                            array_creator.df.columns)
                array_creator.scaler[f'{self.mode}_{self.iter}']['MinMaxScaler'] = MinMaxScaler()
                array_creator.scaler[f'{self.mode}_{self.iter}']['MinMaxScaler'].fit(
                    array_creator.df.iloc[:, instructions['parameters']['MinMaxScaler']].to_numpy().reshape(-1, 1))

            if 'StandardScaler' in options.keys():
                instructions['parameters']['StandardScaler'] = str_to_list(options['StandardScaler'],
                                                                            array_creator.df.columns)
                array_creator.scaler[f'{self.mode}_{self.iter}']['StandardScaler'] = StandardScaler()
                array_creator.scaler[f'{self.mode}_{self.iter}']['StandardScaler'].fit(
                    array_creator.df.iloc[:, instructions['parameters']['StandardScaler']].to_numpy().reshape(-1, 1))

        if 'Categorical' in options.keys():
            instructions['parameters']['Categorical'] = {}
            instructions['parameters']['Categorical']['lst_cols'] = str_to_list(options['Categorical'],
                                                                                array_creator.df.columns)
            for i in instructions['parameters']['Categorical']['lst_cols']:
                instructions['parameters']['Categorical'][f'col_{i}'] = np.unique(
                    array_creator.df.iloc[:, i]).tolist()

        if 'Categorical_ranges' in options.keys():
            self.minvalues = {}
            self.maxvalues = {}

            instructions['parameters']['Categorical_ranges'] = {}
            instructions['parameters']['Categorical_ranges']['lst_cols'] = str_to_list(
                options['Categorical_ranges']['cols'], array_creator.df.columns)
            for i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                self.minvalues[f'col_{i + 1}'] = array_creator.df.iloc[:, i].min()
                self.maxvalues[f'col_{i + 1}'] = array_creator.df.iloc[:, i].max()
                instructions['parameters']['Categorical_ranges'][f'col_{i}'] = {}
                if f'auto_ranges_{i + 1}' in options['Categorical_ranges']:
                    for j in range(int(options['Categorical_ranges'][f'auto_ranges_{i + 1}'])):
                        if (j + 1) == int(options['Categorical_ranges'][f'auto_ranges_{i + 1}']):
                            instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = \
                                array_creator.df.iloc[:, i].max()
                        else:
                            instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = ((
                                                array_creator.df.iloc[:, i].max() - array_creator.df.iloc[
                                                            :, i].min()) / int(
                                options['Categorical_ranges'][f'auto_ranges_{i + 1}'])) * (j + 1)
                else:
                    for j in range(len(options['Categorical_ranges'][f'col_{i + 1}'].split(' '))):
                        instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = float(
                            options['Categorical_ranges'][f'col_{i + 1}'].split(' ')[j])

        if 'one_hot_encoding' in options.keys():
            instructions['parameters']['one_hot_encoding'] = {}
            instructions['parameters']['one_hot_encoding']['lst_cols'] = str_to_list(options['one_hot_encoding'],
                                                                                        array_creator.df.columns)
            for i in instructions['parameters']['one_hot_encoding']['lst_cols']:
                if i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        instructions['parameters']['Categorical_ranges'][f'col_{i}'])
                else:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        np.unique(array_creator.df.iloc[:, i]))

        instructions['instructions'] = np.arange(0, stop, step).tolist()
        instructions['parameters']['put'] = f'{self.mode}_{self.iter}'
        array_creator.df = np.array(array_creator.df)
        return instructions

    def instructions_timeseries(self, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                **options: Параметры временного ряда:
                    lengh: количество примеров для обучения
                    scaler: скейлер
                    y_cols: колонки для предсказания
                    depth: количество значений для предсказания
                    file_name: имя файла.csv
            Returns:
                instructions: dict      Словарь с инструкциями для create_timeseries.
        """
        instructions = {'parameters': {}}
        instructions['parameters']['lengh'] = int(options['lengh'])
        instructions['parameters']['scaler'] = options['scaler']
        instructions['parameters']['y_cols'] = options['file_info']['cols_name'][0]
        instructions['parameters']['depth'] = int(options['depth'])
        step = int(options['step'])
        for key, value in self.tags.items():
            if value == 'dataframe':
                transpose = self.user_parameters[key]['file_info']['transpose']

        if transpose:
            tmp_df_ts = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                   sep=options['file_info']['separator']).T
            tmp_df_ts.columns = tmp_df_ts.iloc[0]
            tmp_df_ts.drop(tmp_df_ts.index[[0]], inplace=True)
            array_creator.y_subdf = tmp_df_ts.loc[:, instructions['parameters']['y_cols'].split(' ')].values
        else:
            array_creator.y_subdf = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                                   sep=options['file_info']['separator'],
                                                   usecols=instructions['parameters']['y_cols'].split(' ')).values

        instructions['parameters']['put'] = f'{self.mode}_{self.iter}'
        instructions['instructions'] = np.arange(0, (len(array_creator.y_subdf) - 1 -
                                                     instructions['parameters']['lengh'] -
                                                     instructions['parameters']['depth']), step).tolist()

        if 'MinMaxScaler' in instructions['parameters'].values():
            array_creator.scaler[f'{self.mode}_{self.iter}'] = MinMaxScaler()
            array_creator.scaler[f'{self.mode}_{self.iter}'].fit(
                array_creator.y_subdf.reshape(-1, 1))

        elif 'StandardScaler' in instructions['parameters'].values():
            array_creator.scaler[f'{self.mode}_{self.iter}'] = StandardScaler()
            array_creator.scaler[f'{self.mode}_{self.iter}'].fit(
                array_creator.y_subdf.reshape(-1, 1))

        return instructions

    def instructions_classification(self, put_data: Union[CreationInputData, CreationOutputData]):
        options = put_data.parameters.native()
        self.one_hot_encoding[put_data.id] = options['one_hot_encoding']
        instructions: dict = {}
        self.task_type[put_data.id] = put_data.type

        if options['file_info']['path_type'] == 'path_file' and not self.y_cls:
            transpose = self.user_parameters.get(1).transpose
            bool_trend = self.user_parameters.get(1).bool_trend
            trend_limit = self.user_parameters.get(1).trend_limit
            lengh = self.user_parameters.get(1).lengh
            step = self.user_parameters.get(1).step
            # for key, value in self.tags.items():
            #     if value == 'dataframe':
                    # transpose = self.user_parameters[key]['file_info']['transpose']
                    # try:
                    #     bool_trend = self.user_parameters[key]['trend']
                    #     self.trend_limit = self.user_parameters[key]['trend_limit']
                    #     lengh = int(self.user_parameters[key]['lengh'])
                    #     step = int(self.user_parameters[key]['step'])
                    # except:
                    #     pass
            if bool_trend:
                if transpose:
                    tmp_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                            sep=options['file_info']['separator']).T
                    tmp_df.columns = tmp_df.iloc[0]
                    tmp_df.drop(tmp_df.index[[0]], inplace=True)
                    trend_subdf = tmp_df.loc[:, options['file_info']['cols_name'][0].split(' ')].values
                else:
                    trend_subdf = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]),
                                          sep=options['file_info']['separator'],
                                          usecols=options['file_info']['cols_name']).values
                self.y_cls = []

                for i in range(0, len(trend_subdf) - 1 - lengh, step):
                    if '%' in trend_limit:
                        trend_limit = float(trend_limit[:trend_limit.find('%')])
                        if abs((trend_subdf[i + lengh + 1] - trend_subdf[i]) /
                               trend_subdf[i]) * 100 <= trend_limit:
                            self.y_cls.append(0)
                        elif trend_subdf[i + lengh + 1] > trend_subdf[i]:
                            self.y_cls.append(1)
                        else:
                            self.y_cls.append(2)
                    else:
                        trend_limit = float(trend_limit)
                        if abs(trend_subdf[i + lengh + 1] - trend_subdf[i]) <= trend_limit:
                            self.y_cls.append(0)
                        elif trend_subdf[i + lengh + 1] > trend_subdf[i]:
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
                if 'categorical' in options.keys():
                    for file_name in options['file_info']['path']:
                        if transpose:
                            tmp_df = pd.read_csv(os.path.join(self.file_folder, file_name),
                                                 sep=options['file_info']['separator']).T
                            tmp_df.columns = tmp_df.iloc[0]
                            tmp_df.drop(tmp_df.index[[0]], inplace=True)
                            data = tmp_df.loc[:, options['file_info']['cols_name'][0].split(' ')]
                        else:
                            data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                           usecols=options['file_info']['cols_name'],
                                           sep=options['file_info']['separator'])
                        column = data[options['file_info']['cols_name'][0]].to_list()
                        classes_names = []
                        for elem in column:
                            if elem not in classes_names:
                                classes_names.append(elem)
                        self.classes_names[put_data.id] = classes_names
                        self.num_classes[put_data.id] = len(classes_names)
                        for elem in column:
                            self.y_cls.append(classes_names.index(elem))
                elif 'categorical_ranges' in options.keys():
                    file_name = options['file_info']['path'][0]
                    if transpose:
                        tmp_df = pd.read_csv(os.path.join(self.file_folder, file_name),
                                             sep=options['file_info']['separator']).T
                        tmp_df.columns = tmp_df.iloc[0]
                        tmp_df.drop(tmp_df.index[[0]], inplace=True)
                        data = tmp_df.loc[:, options['file_info']['cols_name'][0].split(' ')]
                    else:
                        data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                       usecols=options['file_info']['cols_name'],
                                           sep=options['file_info']['separator'])
                    column = data[options['file_info']['cols_name'][0]].to_list()
                    self.minvalue_y = min(column)
                    self.maxvalue_y = max(column)
                    if 'auto_ranges' in options['categorical_ranges'].keys():
                        border = max(column) / int(options['categorical_ranges']['auto_ranges'])
                        self.classes_names[put_data.id] = np.linspace(
                            border, self.maxvalue_y, int(options['categorical_ranges']['auto_ranges'])).tolist()
                    else:
                        self.classes_names[put_data.id] = options['categorical_ranges'][
                            'ranges'].split(' ')

                    self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

                    for elem in column:
                        for i in range(len(self.classes_names[put_data.id])):
                            if elem <= int(self.classes_names[put_data.id][i]):
                                self.y_cls.append(i)
                                break
        elif options['file_info']['path_type'] == 'path_folder' and not self.y_cls:
            if ('image' or 'text' or 'audio' or 'video') in self.tags.values() and not self.y_cls:
                self.classes_names[put_data.id] = \
                    sorted(self.user_parameters[2].file_info.path)
                self.num_classes[put_data.id] = len(
                    self.classes_names[put_data.id])
        else:
            self.classes_names[put_data.id] = np.unique(self.y_cls)
            self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

        instructions['parameters'] = {'num_classes': self.num_classes[put_data.id],
                                      'one_hot_encoding': options['one_hot_encoding']}
        instructions['instructions'] = self.y_cls

        return instructions

    def instructions_regression(self, put_data: Union[CreationInputData, CreationOutputData]):
        options = put_data.parameters.native()
        instructions: dict = {}
        instr: list = []

        self.one_hot_encoding[put_data.id] = False
        self.task_type[put_data.id] = put_data.type

        for file_name in options['file_info']['path']:
            data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['file_info']['cols_name'])
            instr = data[options['file_info']['cols_name'][0]].to_list()

        if 'scaler' in options.keys():
            if options['scaler'] == 'MinMaxScaler':
                array_creator.scaler[put_data.id] = MinMaxScaler()
            if options['scaler'] == 'StandardScaler':
                array_creator.scaler[put_data.id] = StandardScaler()
            array_creator.scaler[put_data.id].fit(np.array(instr).reshape(-1, 1))

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_segmentation(self, put_data: Union[CreationInputData, CreationOutputData]):
        options = put_data.parameters.native()
        folder_name = options.get('folder_name', '')
        instr: list = []

        self.classes_names[put_data.id] = options['classes_names']
        self.classes_colors[put_data.id] = options['classes_colors']
        self.num_classes[put_data.id] = len(options['classes_names'])
        self.one_hot_encoding[put_data.id] = True
        self.task_type[put_data.id] = put_data.type

        if options['file_info']['path_type'] == 'path_folder':
            for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['file_info']['path'][0]))):
                instr.append(os.path.join(options['file_info']['path'][0], file_name))
        elif options['file_info']['path_type'] == 'path_file':
            for file_name in options['file_info']['path']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['file_info']['cols_name'])
                instr = data[options['file_info']['cols_name'][0]].to_list()

        instructions = {'instructions': instr,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': len(options['classes_names']),
                                       'shape': (self.user_parameters.get(1).height,
                                                 self.user_parameters.get(1).width),
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
            self.classes_names[f'{self.mode}_{self.iter + i}'] = class_names
            self.num_classes[f'{self.mode}_{self.iter + i}'] = int(data['classes'])

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
