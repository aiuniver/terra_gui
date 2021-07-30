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
import decamelize
from pydantic import DirectoryPath
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import imgaug.augmenters as iaa

from tqdm.notebook import tqdm
from io import open as io_open
from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone

# from terra_ai import out_exchange
from .data import DataType, Preprocesses, PathsData, InstructionsData, DatasetInstructionsData
from . import array_creator
from . import loading as dataset_loading
from ..data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList, CreationInputData, \
    CreationOutputData
from ..data.datasets.dataset import DatasetData


class CreateDTS(object):

    def __init__(self):

        self.dataset_user_data: CreationData
        self.paths: PathsData

        self._datatype = 'DIM'

        self.trds_path: str = ''
        self.file_folder: str = ''

        self.name: str = ''
        self.source: str = ''
        self.tags: dict = {}
        self.user_tags: list = []
        self.language: str = ''

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

        dataset_loading.load(strict_object=strict_object)

        self.zip_params = json.loads(strict_object.json())

    def set_dataset_data(self, layer: Union[CreationInputData, CreationOutputData]):
        self.tags[layer.id] = decamelize.convert(layer.type)
        if isinstance(layer, CreationInputData):
            self.input_names[layer.id] = layer.name
        else:
            self.output_names[layer.id] = layer.name
        self.user_parameters[layer.id] = layer.parameters

    def set_paths(self, data: CreationData):
        dataset_path = os.path.join(data.datasets_path, f'dataset {data.name}')
        instructions_path = None
        arrays_path = os.path.join(dataset_path, "arrays")
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(arrays_path, exist_ok=True)
        if data.use_generator:
            instructions_path = os.path.join(dataset_path, "instructions")
            os.makedirs(instructions_path, exist_ok=True)
        self.paths = PathsData(datasets=dataset_path, instructions=instructions_path, arrays=arrays_path)

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]):
        self.iter = 0
        self.mode = "input" if isinstance(data, CreationInputsList) else "output"
        instructions = {}
        for elem in data:
            self.set_dataset_data(elem)
            self.iter += 1
            instructions_data = InstructionsData(**getattr(self, f"instructions_{decamelize.convert(elem.type)}"
                                                           )(**elem.parameters.dict()))
            instructions.update([(elem.id, instructions_data)])
        return instructions

    def create_instructions(self, creation_data: CreationData):
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

    def create_dataset(self, creation_data: CreationData):
        self.dataset_user_data = creation_data

        self.name = creation_data.name
        self.user_tags = creation_data.tags
        self.use_generator = creation_data.use_generator
        self.trds_path = creation_data.datasets_path
        self.file_folder = str(creation_data.source_path)

        self.source = 'custom dataset'

        self.set_paths(data=creation_data)

        # Создаем инструкции
        self.instructions = self.create_instructions(creation_data)

        # Получаем входные параметры
        for key in self.instructions.inputs.keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                creation_data.source_path,
                self.instructions.inputs[key].instructions[0],
                **self.instructions.inputs[key].parameters
            )
            self.input_shape[key] = array.shape
            self.input_dtype[key] = str(array.dtype)
            self.datatype = array.shape
            self.input_datatype[key] = self.datatype
        # Получаем выходные параметры
        for key in self.instructions.outputs.keys():
            array = getattr(array_creator, f'create_{self.tags[key]}')(
                creation_data.source_path,
                self.instructions.outputs[key].instructions[0],
                **self.instructions.outputs[key].parameters
            )
            if isinstance(array, tuple):
                for i in range(len(array)):
                    self.output_shape[key + i] = array[i].shape
                    self.output_dtype[key + i] = str(array[i].dtype)
                    self.datatype = array[i].shape
                    self.output_datatype[key + i] = self.datatype
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

        self.limit: int = len(self.instructions.inputs[1].instructions)

        data = {}
        if creation_data.use_generator:
            # Сохранение датасета для генератора
            data['zip_params'] = self.zip_params
            for key in self.instructions.keys():
                os.makedirs(os.path.join(self.paths.instructions, key), exist_ok=True)
                for inp in self.instructions[key].keys():
                    with open(os.path.join(self.paths.instructions, key, f'{inp}.json'),
                              'w') as instruction:
                        json.dump(self.instructions[key][inp], instruction)
            with open(os.path.join(self.paths.instructions, 'sequence.json'),
                      'w') as seq:
                json.dump(self.split_sequence, seq)
            if 'text' in self.tags.keys():  # if 'txt_list' in self.createarray.__dict__.keys():
                with open(os.path.join(self.paths.instructions, 'txt_list.json'),
                          'w') as fp:
                    json.dump(array_creator.txt_list, fp)
        else:
            # Сохранение датасета с NumPy
            for key in self.instructions.inputs.keys():
                x: list = []
                for i in range(self.limit):
                    x.append(getattr(array_creator, f"create_{self.tags[key]}")(
                        creation_data.source_path,
                        self.instructions.inputs[key].instructions[i],
                        **self.instructions.inputs[key].parameters))
                self.X['train'][key] = np.array(x)[self.split_sequence['train']]
                self.X['val'][key] = np.array(x)[self.split_sequence['val']]
                self.X['test'][key] = np.array(x)[self.split_sequence['test']]

            for key in self.instructions.outputs.keys():
                if 'object_detection' in self.tags.values():
                    y_1: list = []
                    y_2: list = []
                    y_3: list = []
                    for i in range(self.limit):
                        arrays = getattr(array_creator, f"create_{self.tags[key]}")(
                            creation_data.source_path,
                            self.instructions.outputs[key].instructions[i],
                            **self.instructions.outputs[key].parameters)
                        y_1.append(arrays[0])
                        y_2.append(arrays[1])
                        y_3.append(arrays[2])

                    splits = ['train', 'val', 'test']
                    for spl_seq in splits:
                        for i in range(len(splits)):
                            self.Y[spl_seq][key] = np.array(y_1)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key + 1] = np.array(y_2)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key + 2] = np.array(y_3)[
                                self.split_sequence[spl_seq]]
                else:
                    y: list = []
                    for i in range(self.limit):
                        y.append(getattr(array_creator, f"create_{self.tags[key]}")(
                            creation_data.source_path,
                            self.instructions.outputs[key].instructions[i],
                            **self.instructions.outputs[key].parameters))
                    self.Y['train'][key] = np.array(y)[self.split_sequence['train']]
                    self.Y['val'][key] = np.array(y)[self.split_sequence['val']]
                    self.Y['test'][key] = np.array(y)[self.split_sequence['test']]

            for sample in self.X.keys():
                # os.makedirs(os.path.join(self.trds_path, 'arrays', sample), exist_ok=True)
                for inp in self.X[sample].keys():
                    os.makedirs(os.path.join(self.paths.arrays, sample), exist_ok=True)
                    joblib.dump(self.X[sample][inp],
                                os.path.join(self.paths.arrays, sample, f'{inp}.gz'))

            for sample in self.Y.keys():
                for inp in self.Y[sample].keys():
                    os.makedirs(os.path.join(self.paths.arrays, sample), exist_ok=True)
                    joblib.dump(self.Y[sample][inp],
                                os.path.join(self.paths.arrays, sample, f'{inp}.gz'))

        self.write_preprocesses_to_files()

        attributes = ['name', 'source', 'tags', 'user_tags', 'language',
                      'input_datatype', 'input_dtype', 'input_shape', 'input_names',
                      'output_datatype', 'output_dtype', 'output_shape', 'output_names',
                      'num_classes', 'classes_names', 'classes_colors',
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
        print(data)
        output = DatasetData(**data)
        print(output)
        return output

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

    def instructions_image(self, **options):

        instructions: dict = {}
        instr: list = []
        y_cls: list = []
        cls_idx = 0
        peg_idx = 0
        self.peg.append(0)
        options['put'] = f'{self.mode}_{self.iter}'
        if 'object_detection' in self.tags.values():
            options['object_detection'] = True
        if options['file_info']['path_type'] == 'path_folder':
            for folder_name in options['file_info']['path']:
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
            self.y_cls = y_cls
        elif options['file_info']['path_type'] == 'path_file':
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

    def instructions_video(self, **options: Any):
        folder = options.get('folder_name', '')
        instructions: dict = {}
        class_mode = False
        max_frames = None

        if options.get('class_mode', '') == 'По каждому кадру':
            class_mode = True
            max_frames = options.get('max_frames', '')

        instructions['instructions'] = self.instructions_images_video(folder, class_mode, max_frames)
        instructions['parameters'] = options

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

    def instructions_dataframe(self, **options):
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
                    one_hot_encoding: строка номеров колонок для перевода категорий в ОНЕ
                    file_name: имя файла.csv
                    y_col: столбец датафрейма для классификации
            Returns:
                instructions: dict      Словарь с инструкциями для create_dataframe.
        """

        def str_to_list(str_numbers, df_cols):
            """
            Получает строку из пользовательских номеров колонок,
            возвращает лист индексов данных колонок
            """
            merged = []
            try:
                str_numbers = str_numbers.split(' ')
            except:
                print('Разделите номера колонок ТОЛЬКО пробелами')
            for i in range(len(str_numbers)):
                if '-' in str_numbers[i]:
                    idx = str_numbers[i].index('-')
                    fi = int(str_numbers[i][:idx]) - 1
                    si = int(str_numbers[i][idx + 1:])
                    tmp = list(range(fi, si))
                    merged.extend(tmp)
                elif re.findall(r'\D', str_numbers[i]) != []:
                    merged.append(df_cols.to_list().index(str_numbers[i]))
                else:
                    merged.append(int(str_numbers[i]) - 1)

            return merged

        general_df = pd.read_csv(os.path.join(self.file_folder, options['file_info']['path'][0]), nrows=1)
        array_creator.df_with_y = pd.read_csv(
            os.path.join(self.file_folder, options['file_info']['path'][0]), usecols=(str_to_list(
                options['file_info']['cols_name'][0], general_df.columns) + str_to_list(options['y_col'],
                                                                                        general_df.columns)))
        array_creator.df_with_y.sort_values(by=options['y_col'], inplace=True, ignore_index=True)

        self.peg.append(0)
        for i in range(len(array_creator.df_with_y.loc[:, options['y_col']]) - 1):
            if array_creator.df_with_y.loc[:, options['y_col']][i] != \
                    array_creator.df_with_y.loc[:, options['y_col']][i + 1]:
                self.peg.append(i + 1)
        self.peg.append(len(array_creator.df_with_y))

        array_creator.df = array_creator.df_with_y.iloc[:, str_to_list(
            options['file_info']['cols_name'][0], array_creator.df_with_y.columns)]

        instructions = {'instructions': np.arange(0, len(array_creator.df)).tolist(),
                        'parameters': {'put': f'{self.mode}_{self.iter}'}}

        if 'MinMaxScaler' or 'StandardScaler' in options.keys():
            array_creator.scaler[f'{self.mode}_{self.iter}'] = {}
            if 'MinMaxScaler' in options.keys():
                instructions['parameters']['MinMaxScaler'] = str_to_list(str_numbers=options['MinMaxScaler'],
                                                                         df_cols=array_creator.df.columns)
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
            instructions['parameters']['Categorical_ranges'] = {}
            instructions['parameters']['Categorical_ranges']['lst_cols'] = str_to_list(
                options['Categorical_ranges']['cols'], array_creator.df.columns)
            for i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                instructions['parameters']['Categorical_ranges'][f'col_{i}'] = {}
                for j in range(len(options['Categorical_ranges'][f'col_{i + 1}'].split(' '))):
                    instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = int(
                        options['Categorical_ranges'][f'col_{i + 1}'].split(' ')[j])

        if 'one_hot_encoding' in options.keys():
            instructions['parameters']['one_hot_encoding'] = {}
            instructions['parameters']['one_hot_encoding']['lst_cols'] = str_to_list(options['one_hot_encoding'],
                                                                                     array_creator.df.columns)
            for i in instructions['parameters']['one_hot_encoding']['lst_cols']:
                if i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        options['Categorical_ranges'][f'col_{i + 1}'].split(' '))
                else:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        np.unique(array_creator.df.iloc[:, i]))

        return instructions

    def instructions_classification(self, **options):

        instructions: dict = {}
        self.task_type[f'{self.mode}_{self.iter}'] = 'classification'
        self.one_hot_encoding[f'{self.mode}_{self.iter}'] = options['one_hot_encoding']

        if options['file_info']['path_type'] == 'path_file':
            for file_name in options['file_info']['path']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['file_info']['cols_name'])
                column = data[options['file_info']['cols_name'][0]].to_list()
                classes_names = []
                for elem in column:
                    if elem not in classes_names:
                        classes_names.append(elem)
                self.classes_names[f'{self.mode}_{self.iter}'] = classes_names
                self.num_classes[f'{self.mode}_{self.iter}'] = len(classes_names)
                for elem in column:
                    self.y_cls.append(classes_names.index(elem))

        else:
            for key, value in self.tags.items():
                if value in ['images', 'text', 'audio', 'video']:
                    self.classes_names[f'{self.mode}_{self.iter}'] = \
                        sorted(self.user_parameters[key]['file_info']['path'])
                    self.num_classes[f'{self.mode}_{self.iter}'] = len(self.classes_names[f'{self.mode}_{self.iter}'])

        instructions['parameters'] = {'num_classes': len(np.unique(self.y_cls)),
                                      'one_hot_encoding': options['one_hot_encoding']}
        instructions['instructions'] = self.y_cls

        return instructions

    def instructions_regression(self, **options):

        instructions: dict = {}
        instr: list = []

        self.one_hot_encoding[f'{self.mode}_{self.iter}'] = False
        self.task_type[f'{self.mode}_{self.iter}'] = 'regression'

        for file_name in options['file_info']['path']:
            data = pd.read_csv(os.path.join(self.file_folder, file_name), usecols=options['file_info']['cols_name'])
            instr = data[options['file_info']['cols_name'][0]].to_list()

        if 'scaler' in options.keys():
            if options['scaler'] == 'MinMaxScaler':
                array_creator.scaler[f'{self.mode}_{self.iter}'] = MinMaxScaler()
            if options['scaler'] == 'StandardScaler':
                array_creator.scaler[f'{self.mode}_{self.iter}'] = StandardScaler()
            array_creator.scaler[f'{self.mode}_{self.iter}'].fit(np.array(instr).reshape(-1, 1))

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_segmentation(self, **options):
        print(options)
        folder_name = options.get('folder_name', '')
        instr: list = []

        self.classes_names[f'{self.mode}_{self.iter}'] = options['classes_names']
        self.classes_colors[f'{self.mode}_{self.iter}'] = options['classes_colors']
        self.num_classes[f'{self.mode}_{self.iter}'] = len(options['classes_names'])
        self.one_hot_encoding[f'{self.mode}_{self.iter}'] = True
        self.task_type[f'{self.mode}_{self.iter}'] = 'segmentation'

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
