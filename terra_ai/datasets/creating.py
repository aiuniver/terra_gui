import os
import random
from typing import Union

import numpy as np
import pandas as pd
import re
import pymorphy2
# import shutil
import json
import joblib
# from pydantic import DirectoryPath
# from pydantic.types import FilePath
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from librosa import load as librosa_load
import imgaug.augmenters as iaa

# from tqdm.notebook import tqdm
# from io import open as io_open
# from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone
import cv2

# from terra_ai import out_exchange
from ..data.datasets.creations.layers.input.types.Text import TextModeChoice
from ..data.datasets.creations.layers.input.types.Audio import AudioModeChoice
from ..data.datasets.creations.layers.input.types.Video import VideoModeChoice
from ..data.datasets.extra import DatasetGroupChoice  # LayerScalerChoice
from ..utils import decamelize
from .data import DataType, Preprocesses, PathsData, InstructionsData, DatasetInstructionsData
from . import array_creator
# from . import loading as dataset_loading
from ..data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList, CreationInputData, \
    CreationOutputData, OneFileData
from ..data.datasets.dataset import DatasetData, DatasetInputsData, DatasetOutputsData  # DatasetLayerData


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
        self.put_tags: dict = {'inputs': {}, 'outputs': {}}
        self.user_tags: list = []
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
        self.tsgenerator: dict = {}
        self.temporary: dict = {}
        self.minvalue_y: int = 0
        self.maxvalue_y: int = 0

        self.minvalues: dict = {}
        self.maxvalues: dict = {}
        self.instructions = None
        self.paths = None
        self.dataset_user_data = None
        self.build_dataframe: dict = {}
        self.dataframe: dict = {}

    def create_one_file_array(self, file_data: OneFileData, source_path):
        instructions_data = getattr(self, f"instructions_{decamelize(file_data.type)}")(file_data)
        array = getattr(array_creator, f"create_{decamelize(file_data.type)}")(
            source_path,
            instructions_data.get("instructions")[0],
            **instructions_data.get("parameters"))
        return array

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

        self.create_table(creation_data=creation_data)

        # Получаем входные параметры
        self.inputs = self.create_inputs_parameters(creation_data=creation_data)

        # Получаем выходные параметры
        self.outputs = self.create_output_parameters(creation_data=creation_data)

        for key in self.dataframe.keys():
            self.dataframe[key].to_csv(os.path.join(self.paths.instructions, f'{key}.csv'))

        # if creation_data.use_generator:
        #     # Сохранение датасета для генератора
        #     with open(os.path.join(self.paths.instructions, f'generator_instructions.json'),
        #               'w') as instruction:
        #         json.dump(self.instructions.native(), instruction)
        #     with open(os.path.join(self.paths.instructions, 'sequence.json'), 'w') as seq:
        #         json.dump(self.split_sequence, seq)
        #     if 'text' in self.tags.keys():  # if 'txt_list' in self.createarray.__dict__.keys():
        #         with open(os.path.join(self.paths.instructions, 'txt_list.json'), 'w') as fp:
        #             json.dump(array_creator.txt_list, fp)
        # else:
        if not creation_data.use_generator:
            # Сохранение датасета с NumPy
            x_array = self.create_dataset_arrays(creation_data=creation_data, put_data=self.instructions.inputs)
            y_array = self.create_dataset_arrays(creation_data=creation_data, put_data=self.instructions.outputs)

            self.write_arrays(x_array, y_array)

        # запись препроцессов (скейлер, токенайзер и т.п.)
        self.write_preprocesses_to_files()

        # запись параметров в json
        self.write_instructions_to_files()

        # создание и запись конфигурации датасета
        output = DatasetData(**self.create_dataset_configure(creation_data=creation_data))

        # print(output.json(indent=2))
        return output

    def set_dataset_data(self, layer: Union[CreationInputData, CreationOutputData]):
        self.tags[layer.id] = decamelize(layer.type)
        if isinstance(layer, CreationInputData):
            self.input_names[layer.id] = layer.name
            self.put_tags['inputs'][layer.id] = decamelize(layer.type)
        else:
            self.output_names[layer.id] = layer.name
            self.put_tags['outputs'][layer.id] = decamelize(layer.type)
        self.user_parameters[layer.id] = layer.parameters

        pass

    def set_paths(self, data: CreationData) -> PathsData:
        dataset_path = os.path.join(data.datasets_path, f'dataset {data.name}')
        arrays_path = os.path.join(dataset_path, "arrays")
        instructions_path = os.path.join(dataset_path, "instructions")
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(arrays_path, exist_ok=True)
        os.makedirs(instructions_path, exist_ok=True)
        return PathsData(datasets=dataset_path, instructions=instructions_path, arrays=arrays_path)

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:
        inputs = self.create_put_instructions(data=creation_data.inputs)
        outputs = self.create_put_instructions(data=creation_data.outputs)
        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)
        return instructions

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:

        instructions: dict = {}
        for elem in data:
            paths_list: list = []
            if decamelize(elem.type) not in ['text_segmentation']:
                for paths in elem.parameters.sources_paths:
                    if not paths.suffix == '.csv':
                        for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, paths))):
                            if file_name:
                                file_folder = directory.replace(self.file_folder, '')[1:]
                                for name in sorted(file_name):
                                    paths_list.append(os.path.join(file_folder, name))
                    elif paths.suffix == '.csv' and elem.type not in ['Dataframe', 'Timeseries']:
                        data = pd.read_csv(os.path.join(self.file_folder, paths),
                                           usecols=elem.parameters.cols_names)
                        paths_list = data[elem.parameters.cols_names[0]].to_list()
            instructions_data = InstructionsData(**getattr(self, f"instructions_{decamelize(elem.type)}")(paths_list,
                                                                                                          elem))
            instructions.update([(elem.id, instructions_data)])

        return instructions

    def write_instructions_to_files(self):

        os.makedirs(self.paths.instructions, exist_ok=True)
        for put in self.instructions.__dict__.keys():
            for idx in self.instructions.__dict__[put].keys():
                # if decamelize(LayerOutputTypeChoice.ObjectDetection) in self.tags[idx]:
                #     for i in range(6):
                #         with open(os.path.join(self.paths.instructions, f'{idx+i}_{put}.json'), 'w') as cfg:
                #             json.dump(self.instructions.__dict__[put][idx].parameters, cfg)
                # else:
                with open(os.path.join(self.paths.instructions, f'{idx}_{put}.json'), 'w') as cfg:
                    json.dump(self.instructions.__dict__[put][idx].parameters, cfg)
        pass

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
            if self.tags[key] in ['video', 'text', 'audio']:
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                    creation_data.source_path,
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}'],
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}_slice'],
                    **self.instructions.inputs.get(key).parameters
                )
            elif self.tags[key] == 'dataframe':
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                            creation_data.source_path,
                            self.instructions.inputs.get(key).instructions[0],
                            **self.instructions.inputs.get(key).parameters
                        )
            else:
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                    creation_data.source_path,
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}'],
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
            if self.tags[key] in ['text', 'text_segmentation', 'audio']:
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                    creation_data.source_path,
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}'],
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}_slice'],
                    **self.instructions.outputs.get(key).parameters
                )
            elif self.tags[key] == 'timeseries':
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                            creation_data.source_path,
                            self.instructions.outputs.get(key).instructions[0],
                            **self.instructions.outputs.get(key).parameters
                        )
            else:
                array = getattr(array_creator, f'create_{self.tags[key]}')(
                    creation_data.source_path,
                    self.dataframe['train'].loc[0, f'{key}_{self.tags[key]}'],
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

    def create_table(self, creation_data: CreationData):

        split_sequence = {'train': [],
                          'val': [],
                          'test': []}
        
        for i in range(len(self.peg) - 1):
            indices = np.arange(self.peg[i], self.peg[i + 1])
            train_len = int(creation_data.info.part.train * len(indices))
            val_len = int(creation_data.info.part.validation * len(indices))
            indices = indices.tolist()
            split_sequence['train'].extend(indices[:train_len])
            split_sequence['val'].extend(indices[train_len:train_len + val_len])
            split_sequence['test'].extend(indices[train_len + val_len:])
        if creation_data.info.shuffle:
            random.shuffle(split_sequence['train'])
            random.shuffle(split_sequence['val'])
            random.shuffle(split_sequence['test'])

        array_creator.df_ts = pd.DataFrame(self.build_dataframe)
        for key, value in split_sequence.items():
            self.dataframe[key] = array_creator.df_ts.loc[value, :].reset_index(drop=True)

    def create_dataset_arrays(self, creation_data: CreationData, put_data: dict) -> dict:
        out_array = {'train': {}, 'val': {}, 'test': {}}
        splits = list(out_array.keys())
        num_arrays = 1
        for split in splits:
            for key in put_data.keys():
                current_arrays: list = []
                if self.tags[key] == 'object_detection':
                    num_arrays = 6
                    for i in range(num_arrays):
                        globals()[f'current_arrays_{i + 1}'] = []
                for i in range(len(self.dataframe[split])):
                    if self.tags[key] in ['video', 'text', 'audio', 'text_segmentation']:
                        array = getattr(array_creator, f'create_{self.tags[key]}')(
                            creation_data.source_path,
                            self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'],
                            self.dataframe[split].loc[i, f'{key}_{self.tags[key]}_slice'],
                            **put_data.get(key).parameters
                        )
                    elif self.tags[key] in ['dataframe', 'timeseries']:
                        array = getattr(array_creator, f'create_{self.tags[key]}')(
                            creation_data.source_path,
                            put_data.get(key).instructions[i],
                            **put_data.get(key).parameters
                        )
                    else:
                        array = getattr(array_creator, f'create_{self.tags[key]}')(
                            creation_data.source_path,
                            self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'],
                            **put_data.get(key).parameters
                        )

                    if self.tags[key] == 'object_detection':
                        for j in range(num_arrays):
                            globals()[f'current_arrays_{j + 1}'].append(array[j])
                    else:
                        current_arrays.append(array)

                if self.tags[key] == 'object_detection':
                    for j in range(num_arrays):
                        out_array[split][key + j] = np.array(globals()[f'current_arrays_{j + 1}'])
                else:
                    out_array[split][key] = np.array(current_arrays)
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
                      'encoding', 'task_type', 'put_tags', 'use_generator']

        size_bytes = 0
        for path, dirs, files in os.walk(os.path.join(self.trds_path, f'dataset {self.name}')):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        tags_list = []
        for value in self.tags.values():
            tags_list.append({'alias': value, 'name': value.title()})
        self.tags = tags_list

        for attr in attributes:
            data[attr] = self.__dict__[attr]
        data['date'] = datetime.now().astimezone(timezone('Europe/Moscow')).isoformat()
        data['alias'] = creation_data.alias
        data['size'] = {'value': size_bytes}

        data['group'] = DatasetGroupChoice.custom
        with open(os.path.join(self.trds_path, f'dataset {self.name}', 'config.json'), 'w') as fp:
            json.dump(data, fp)

        return data

    def instructions_image(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData, OneFileData]):

        instructions: dict = {}
        self.peg.append(0)
        options = put_data.parameters.native()
        options['put'] = put_data.id
        if 'object_detection' in self.tags.values():
            options['object_detection'] = True
            for path in paths_list:
                if path.endswith('.txt'):
                    paths_list.remove(path)

        for key, value in self.tags.items():
            if value == 'classification':
                if self.user_parameters[key].sources_paths and\
                        os.path.isfile(self.user_parameters[key].sources_paths[0]):
                    data = pd.read_csv(self.user_parameters[key].sources_paths[0],
                                       usecols=self.user_parameters[key].cols_names)
                    self.y_cls = data[self.user_parameters[key].cols_names[0]].to_list()
                else:
                    peg_idx = 0
                    prev_class = paths_list[0].split(os.path.sep)[-2]
                    for elem in paths_list:
                        cur_class = elem.split(os.path.sep)[-2]
                        if cur_class != prev_class:
                            prev_class = cur_class
                            self.peg.append(peg_idx)
                        self.y_cls.append(cur_class)
                        peg_idx += 1
                break
        self.peg.append(len(paths_list))

        if options.get('augmentation', None):
            aug_parameters = []
            for key, value in options['augmentation'].items():
                aug_parameters.append(getattr(iaa, key)(**value))
            array_creator.augmentation[put_data.id] = iaa.Sequential(aug_parameters, random_order=True)
            del options['augmentation']

        instructions['parameters'] = options
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_image': paths_list}
        else:
            self.build_dataframe[f'{put_data.id}_image'] = paths_list

        return instructions

    def instructions_video(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                paths_list: list
                    Путь к файлам.
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
        instructions: dict = {}
        video: list = []
        video_slice: list = []
        y_cls: list = []
        csv_y_cls: list = []
        peg_idx = 0
        csv_flag = False
        self.peg.append(0)

        for key, value in self.tags.items():
            if value == 'classification':
                if self.user_parameters[key].sources_paths and\
                        os.path.isfile(self.user_parameters[key].sources_paths[0]):
                    data = pd.read_csv(self.user_parameters[key].sources_paths[0],
                                       usecols=self.user_parameters[key].cols_names)
                    csv_y_cls = data[self.user_parameters[key].cols_names[0]].to_list()
                    csv_flag = True

        prev_class = paths_list[0].split(os.path.sep)[-2]
        for idx, elem in enumerate(paths_list):
            cur_class = elem.split(os.path.sep)[-2]
            if options['video_mode'] == VideoModeChoice.completely:
                video.append(elem)
                video_slice.append([0, options['max_frames']])
                peg_idx += 1
                if cur_class != prev_class:
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                y_cls.append(cur_class)
            elif options['video_mode'] == VideoModeChoice.length_and_step:
                cur_step = 0
                stop_flag = False
                cap = cv2.VideoCapture(os.path.join(self.file_folder, elem))
                frame_count = int(cap.get(7))
                while not stop_flag:
                    video.append(elem)
                    video_slice.append([cur_step, cur_step + options['length']])
                    peg_idx += 1
                    if cur_class != prev_class:
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    y_cls.append(csv_y_cls[idx]) if csv_flag else y_cls.append(cur_class)
                    cur_step += options['step']
                    if cur_step + options['length'] > frame_count:
                        stop_flag = True
                        if options['length'] < frame_count:
                            video.append(elem)
                            video_slice.append([frame_count - options['length'], frame_count])
                            y_cls.append(csv_y_cls[idx]) if csv_flag else y_cls.append(cur_class)

        self.y_cls = y_cls
        self.peg.append(len(video))

        del options['video_mode']
        del options['length']
        del options['step']
        del options['max_frames']
        options['put'] = put_data.id

        instructions['parameters'] = options
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_video': video,
                                            f'{put_data.id}_video_slice': video_slice}
        else:
            self.build_dataframe[f'{put_data.id}_image'] = paths_list

        return instructions

    def instructions_text(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData]):

        def read_text(file_path, lower, del_symbols, split, open_symbol=None, close_symbol=None) -> str:

            with open(os.path.join(self.file_folder, file_path), 'r', encoding='utf-8') as txt:
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
        filters = options.get('filters', '')
        instructions: dict = {}
        txt_list: dict = {}
        lower: bool = True
        split: str = ' '
        open_symbol = None
        close_symbol = None
        tags = None
        csv_flag = False
        y_cls: list = []
        csv_y_cls: list = []

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

        for key, value in self.tags.items():
            if value == 'classification':
                if self.user_parameters[key].sources_paths and\
                        os.path.isfile(self.user_parameters[key].sources_paths[0]):
                    data = pd.read_csv(self.user_parameters[key].sources_paths[0],
                                       usecols=self.user_parameters[key].cols_names)
                    csv_y_cls = data[self.user_parameters[key].cols_names[0]].to_list()
                    csv_flag = True

        for idx, path in enumerate(paths_list):
            if not csv_flag:
                # if os.path.isfile(os.path.join(self.file_folder, path)):
                txt_list[path] = read_text(path, lower, filters, split, open_symbol, close_symbol)
            else:
                txt_list[str(idx)] = path

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
            array_creator.create_tokenizer(put_data.id, **{'num_words': options.get('max_words_count', int),
                                                           'filters': filters,
                                                           'lower': lower,
                                                           'split': split,
                                                           'char_level': False,
                                                           'oov_token': '<UNK>'})
            array_creator.tokenizer[put_data.id].fit_on_texts(list(txt_list.values()))

        # array_creator.txt_list[put_data.id] = txt_list
        self.temporary[put_data.id] = txt_list

        text: list = []
        text_slice: list = []
        peg_idx: int = 0

        if not csv_flag:
            prev_class: str = sorted(txt_list.keys())[0].split(os.path.sep)[-2]
            self.peg.append(0)
        else:
            prev_class: str = csv_y_cls[0]

        for idx, (key, value) in enumerate(sorted(txt_list.items())):
            if not csv_flag:
                cur_class = key.split(os.path.sep)[-2]
            else:
                cur_class = csv_y_cls[idx]
            if options['text_mode'] == TextModeChoice.completely:
                text.append(value)
                text_slice.append([0, options['max_words']])
                if cur_class != prev_class:
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                peg_idx += 1
                y_cls.append(cur_class)

            elif options['text_mode'] == TextModeChoice.length_and_step:
                max_length = len(value.split(' '))
                if 'text_segmentation' in self.tags.values():
                    count = 0
                    for elem in tags.split(' '):
                        count += value.split(' ').count(elem)
                    max_length -= count
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text.append(' '.join(value.split(' ')[cur_step: cur_step + options['length']]))
                    text_slice.append([cur_step, cur_step + options['length']])
                    peg_idx += 1
                    if cur_class != prev_class:
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    y_cls.append(csv_y_cls[idx]) if csv_flag else y_cls.append(cur_class)
                    cur_step += options['step']
                    if cur_step + options['length'] > max_length:
                        stop_flag = True

        self.peg.append(len(text))
        self.y_cls = y_cls

        instructions['parameters'] = {'embedding': options.get('embedding', bool),
                                      'bag_of_words': options.get('bag_of_words', bool),
                                      'word_to_vec': options.get('word_to_vec', bool),
                                      'put': put_data.id
                                      }
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_text': text,
                                            f'{put_data.id}_text_slice': text_slice}
        else:
            self.build_dataframe[f'{put_data.id}_text'] = text
            self.build_dataframe[f'{put_data.id}_text_slice'] = text_slice

        return instructions

    def instructions_audio(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        sample_rate = options.get('sample_rate', int)
        options = put_data.parameters.native()
        instructions: dict = {}
        audio: list = []
        audio_slice: list = []
        y_cls: list = []
        csv_y_cls: list = []
        peg_idx = 0
        csv_flag = False
        self.peg.append(0) if not self.peg else None

        for key, value in self.tags.items():
            if value == 'classification':
                if self.user_parameters[key].sources_paths and\
                        os.path.isfile(self.user_parameters[key].sources_paths[0]):
                    data = pd.read_csv(self.user_parameters[key].sources_paths[0],
                                       usecols=self.user_parameters[key].cols_names)
                    csv_y_cls = data[self.user_parameters[key].cols_names[0]].to_list()
                    csv_flag = True

        prev_class = paths_list[0].split(os.path.sep)[-2]
        for idx, elem in enumerate(paths_list):
            cur_class = elem.split(os.path.sep)[-2]
            if options['audio_mode'] == AudioModeChoice.completely:
                audio.append(elem)
                audio_slice.append([0, options['max_frames']])
                peg_idx += 1
                if cur_class != prev_class:
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                y_cls.append(cur_class)
            elif options['audio_mode'] == AudioModeChoice.length_and_step:
                cur_step = 0.0
                stop_flag = False
                y, sr = librosa_load(path=os.path.join(self.file_folder, elem), sr=sample_rate, res_type='scipy')
                sample_length = len(y) / sample_rate
                while not stop_flag:
                    audio.append(elem)
                    audio_slice.append([cur_step, cur_step + options['length']])
                    peg_idx += 1
                    if cur_class != prev_class:
                        self.peg.append(peg_idx)
                        prev_class = cur_class
                    y_cls.append(csv_y_cls[idx]) if csv_flag else y_cls.append(cur_class)
                    cur_step += options['step']
                    if cur_step + options['length'] > sample_length:
                        stop_flag = True

        self.y_cls = y_cls
        self.peg.append(len(audio))

        for elem in ['audio_mode', 'file_info', 'length', 'step', 'max_seconds']:
            if elem in options.keys():
                del options[elem]

        instructions['parameters'] = options
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_audio': audio,
                                            f'{put_data.id}_audio_slice': audio_slice}
        else:
            self.build_dataframe[f'{put_data.id}_audio'] = audio
            self.build_dataframe[f'{put_data.id}_audio_slice'] = audio_slice

        return instructions

    def instructions_dataframe(self, _, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                _
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
        instructions = {'instructions': {},
                        'parameters': {}}
        if 'classification' in self.tags.values():
            step = 1
            y_col = self.user_parameters[2].cols_names
            if options['pad_sequences'] or options['xlen_step']:
                if options['pad_sequences']:
                    example_length = int(options['example_length'])
                    if transpose:
                        tmp_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             sep=options['separator']).T
                        tmp_df.columns = tmp_df.iloc[0]
                        tmp_df.drop(tmp_df.index[[0]], inplace=True)
                        df_y = tmp_df.loc[:, list(range(example_length + 1))]
                    else:
                        df_y = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                           usecols=list(range(example_length + 1)),
                                           sep=options['separator'])

                    df_y.fillna(0, inplace=True)
                    df_y.sort_values(by=y_col[0], inplace=True, ignore_index=True)
                    self.peg.append(0)
                    for i in range(len(df_y) - 1):
                        if df_y[y_col[0]][i] != df_y[y_col[0]][i + 1]:
                            self.peg.append(i + 1)
                    self.peg.append(len(df_y))

                    df = df_y.iloc[:, 1:]
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
                    df.sort_values(by=y_col[0], inplace=True, ignore_index=True)
                    df = df.iloc[:, 1:]
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

                    self.peg.append(0)
                    for i in range(len(self.y_cls) - 1):
                        if self.y_cls[i] != self.y_cls[i + 1]:
                            self.peg.append(i + 1)
                    self.peg.append(len(self.y_cls))

                if 'min_max_scaler' in options.values():
                    array_creator.scaler[put_data.id] = MinMaxScaler()
                    array_creator.scaler[put_data.id].fit(
                        df.values.reshape(-1, 1))

                elif 'standard_scaler' in options.values():
                    array_creator.scaler[put_data.id] = StandardScaler()
                    array_creator.scaler[put_data.id].fit(
                        df.values.reshape(-1, 1))

                if options['xlen_step']:
                    df = pd.DataFrame({'slices': xlen_array})
                instructions['parameters']['scaler'] = options['scaler']
                instructions['parameters']['put'] = put_data.id

            else:
                if transpose:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             sep=options['separator']).T
                    general_df.columns = general_df.iloc[0]
                    general_df.drop(general_df.index[[0]], inplace=True)

                    xdf = general_df.iloc[:, str_to_list(options['cols_names'][0], general_df.columns)]
                    ydf = general_df.loc[:, y_col]
                else:
                    general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                             nrows=1,
                                             sep=options['separator'])
                    xdf = pd.read_csv(
                        os.path.join(self.file_folder, options['sources_paths'][0]), usecols=str_to_list(
                            options['cols_names'][0], general_df.columns), sep=options['separator'])
                    ydf = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                      usecols=y_col, sep=options['separator'])

                df_with_y = pd.concat((xdf, ydf), axis=1)
                df_with_y.sort_values(by=y_col[0], inplace=True, ignore_index=True)

                self.peg.append(0)
                for i in range(len(df_with_y.loc[:, y_col[0]]) - 1):
                    if df_with_y.loc[:, y_col[0]][i] != df_with_y.loc[:, y_col[0]][i + 1]:
                        self.peg.append(i + 1)
                self.peg.append(len(df_with_y))

                df = df_with_y.iloc[:, list(range(len(df_with_y.columns) - 1))]
                instructions = {'parameters': {}}
            stop = len(df)
        elif 'timeseries' in self.tags.values():
            bool_trend = self.user_parameters[2].trend
            if bool_trend:
                depth = 1
            else:
                depth = int(self.user_parameters[2].depth)
            length = int(self.user_parameters[2].length)
            step = int(self.user_parameters[2].step)
            if transpose:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator']).T
                general_df.columns = general_df.iloc[0]
                general_df.drop(general_df.index[[0]], inplace=True)
                general_df.index = range(0, len(general_df))
                for i in str_to_list(options['cols_names'][0], general_df.columns):
                    general_df = general_df.astype({general_df.columns[i]: np.float}, errors='ignore')
                df = general_df.iloc[:, str_to_list(options['cols_names'][0],
                                                                  general_df.columns)]
            else:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         nrows=1,
                                         sep=options['separator'])
                df = pd.read_csv(
                    os.path.join(self.file_folder, options['sources_paths'][0]), usecols=(str_to_list(
                        options['cols_names'][0], general_df.columns)),
                    sep=options['separator'])

            stop = len(df) - length - depth
            instructions = {'parameters': {'timeseries': True,
                                           'length': length,
                                           'depth': depth,
                                           'step': step,
                                           'bool_trend': bool_trend}}

            self.peg.append(0)
            self.peg.append(len(np.arange(0, len(df) - length - depth - 1, step)))
        else:
            step = 1
            if transpose:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         sep=options['separator']).T
                general_df.columns = general_df.iloc[0]
                general_df.drop(general_df.index[[0]], inplace=True)
                df = general_df.iloc[:, str_to_list(options['cols_names'][0], general_df.columns)]
            else:
                general_df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                         nrows=1,
                                         sep=options['separator'])
                df = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                               usecols=(str_to_list(options['cols_names'][0], general_df.columns)),
                                               sep=options['separator'])

            self.peg.append(0)
            self.peg.append(len(df))
            instructions = {'parameters': {}}
            stop = len(df)

        if options['MinMaxScaler'] or options['StandardScaler']:
            array_creator.scaler[put_data.id] = {'MinMaxScaler': {},
                                                 'StandardScaler': {}}
            if options['MinMaxScaler']:
                instructions['parameters']['MinMaxScaler'] = str_to_list(options['MinMaxScaler'],
                                                                         df.columns)
                for i in instructions['parameters']['MinMaxScaler']:
                    array_creator.scaler[put_data.id]['MinMaxScaler'][f'col_{i + 1}'] = MinMaxScaler()
                    array_creator.scaler[put_data.id]['MinMaxScaler'][f'col_{i + 1}'].fit(
                        df.iloc[:, [i]].to_numpy().reshape(-1, 1))

            if options['StandardScaler']:
                instructions['parameters']['StandardScaler'] = str_to_list(options['StandardScaler'],
                                                                           df.columns)
                for i in instructions['parameters']['StandardScaler']:
                    array_creator.scaler[put_data.id]['StandardScaler'][f'col_{i + 1}'] = StandardScaler()
                    array_creator.scaler[put_data.id]['StandardScaler'][f'col_{i + 1}'].fit(
                        df.iloc[:, [i]].to_numpy().reshape(-1, 1))

        if options['Categorical']:
            instructions['parameters']['Categorical'] = {}
            instructions['parameters']['Categorical']['lst_cols'] = str_to_list(options['Categorical'],
                                                                                df.columns)
            for i in instructions['parameters']['Categorical']['lst_cols']:
                instructions['parameters']['Categorical'][f'col_{i}'] = list(set(
                    df.iloc[:, i]))

        if options['Categorical_ranges']:
            instructions['parameters']['Categorical_ranges'] = {}
            tmp_lst = str_to_list(options['Categorical_ranges'], df.columns)
            instructions['parameters']['Categorical_ranges']['lst_cols'] = tmp_lst
            for i in range(len(tmp_lst)):
                self.minvalues[f'col_{tmp_lst[i] + 1}'] = df.iloc[:, tmp_lst[i]].min()
                self.maxvalues[f'col_{tmp_lst[i] + 1}'] = df.iloc[:, tmp_lst[i]].max()
                instructions['parameters']['Categorical_ranges'][f'col_{tmp_lst[i]}'] = {}
                if len(list(options['cat_cols'].values())[i].split(' ')) == 1:
                    for j in range(int(list(options['cat_cols'].values())[i])):
                        if (j + 1) == int(list(options['cat_cols'].values())[i]):
                            instructions['parameters']['Categorical_ranges'][f'col_{tmp_lst[i]}'][f'range_{j}'] = \
                                df.iloc[:, tmp_lst[i]].max()
                        else:
                            instructions['parameters']['Categorical_ranges'][f'col_{tmp_lst[i]}'][f'range_{j}'] = \
                                ((df.iloc[:, tmp_lst[i]].max() - df.iloc[:,tmp_lst[i]].min()) / int(
                                    list(options['cat_cols'].values())[i]) * (j + 1))
                else:
                    for j in range(len(list(options['cat_cols'].values())[i].split(' '))):
                        instructions['parameters']['Categorical_ranges'][f'col_{tmp_lst[i]}'][f'range_{j}'] = float(
                            list(options['cat_cols'].values())[i].split(' ')[j])

        if options['one_hot_encoding']:
            instructions['parameters']['one_hot_encoding'] = {}
            instructions['parameters']['one_hot_encoding']['lst_cols'] = str_to_list(options['one_hot_encoding'],
                                                                                     df.columns)
            for i in instructions['parameters']['one_hot_encoding']['lst_cols']:
                if options['Categorical_ranges'] and i in str_to_list(
                        options['Categorical_ranges'], df.columns):
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        instructions['parameters']['Categorical_ranges'][f'col_{i}'])
                else:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        set(df.iloc[:, i]))

        if options['xlen_step']:
            instructions['parameters']['xlen_step'] = True
        else:
            instructions['parameters']['xlen_step'] = False

        array_creator.columns = df.columns
        instructions['instructions'] = np.arange(0, stop, step).tolist()
        for i in df.columns:
            self.build_dataframe.update({i: df.loc[:, i]})
        instructions['parameters']['put'] = put_data.id
        return instructions

    def instructions_timeseries(self, _, put_data: Union[CreationInputData, CreationOutputData]):
        """
            Args:
                _
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
        instructions['parameters']['y_cols'] = options['cols_names'][0]
        bool_trend = options['trend']
        instructions['parameters']['bool_trend'] = bool_trend
        transpose = self.user_parameters.get(
            list(self.tags.keys())[list(self.tags.values()).index('dataframe')]).transpose
        separator = self.user_parameters.get(
            list(self.tags.keys())[list(self.tags.values()).index('dataframe')]).separator
        step = int(options['step'])
        if transpose:
            tmp_df_ts = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                    sep=separator).T
            tmp_df_ts.columns = tmp_df_ts.iloc[0]
            tmp_df_ts.drop(tmp_df_ts.index[[0]], inplace=True)
            tmp_df_ts.index = range(0, len(tmp_df_ts))
            for i in instructions['parameters']['y_cols'].split(' '):
                tmp_df_ts = tmp_df_ts.astype({i: np.float}, errors='ignore')
            y_subdf = tmp_df_ts.loc[:, instructions['parameters']['y_cols'].split(' ')]
        else:
            y_subdf = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                                sep=separator,
                                                usecols=instructions['parameters']['y_cols'].split(' '))
        if bool_trend:
            trend_limit = options['trend_limit']
            if '%' in trend_limit:
                trend_limit = float(trend_limit[:trend_limit.find('%')])
                for i in range(0, len(y_subdf) - instructions['parameters']['length'], step):
                    if abs((y_subdf.iloc[i + instructions['parameters']['length'] + 1][0] -
                            y_subdf.iloc[i][0]) / y_subdf.iloc[i][0]) * 100 <= trend_limit:
                        self.y_cls.append(0)
                    elif y_subdf.iloc[i + instructions['parameters']['length'] + 1][0] > \
                            y_subdf.iloc[i][0]:
                        self.y_cls.append(1)
                    else:
                        self.y_cls.append(2)
            else:
                trend_limit = float(trend_limit)
                for i in range(0, len(y_subdf) - instructions['parameters']['length'], step):
                    if abs(y_subdf.iloc[i + instructions['parameters']['length'] + 1][0]
                           - y_subdf.iloc[i][0]) <= trend_limit:
                        self.y_cls.append(0)
                    elif y_subdf.iloc[i + instructions['parameters']['length'] + 1][0] > \
                            y_subdf.iloc[i][0]:
                        self.y_cls.append(1)
                    else:
                        self.y_cls.append(2)
            #     if options['one_hot_encoding']:
            #         tmp_unique = list(set(self.y_cls))
            #         for i in range(len(self.y_cls)):
            #             self.y_cls[i] = tmp_unique.index(self.y_cls[i])
            self.classes_names[put_data.id] = ['Не изменился', 'Вверх', 'Вниз']
            self.num_classes[put_data.id] = len(self.classes_names[put_data.id])
            instructions['instructions'] = self.y_cls
            self.build_dataframe[y_subdf.columns[0]] = \
                y_subdf[y_subdf.columns[0]]
        else:
            instructions['parameters']['scaler'] = options['scaler']
            instructions['parameters']['depth'] = int(options['depth'])

            if 'min_max_scaler' in instructions['parameters'].values():
                array_creator.scaler[put_data.id] = MinMaxScaler()
                array_creator.scaler[put_data.id].fit(
                    y_subdf.values.reshape(-1, 1))
            elif 'standard_scaler' in instructions['parameters'].values():
                array_creator.scaler[put_data.id] = StandardScaler()
                array_creator.scaler[put_data.id].fit(
                    y_subdf.values.reshape(-1, 1))

            instructions['instructions'] = np.arange(0, (len(y_subdf) -
                                                                    instructions['parameters']['length'] -
                                                                    instructions['parameters']['depth']), step).tolist()
            for i in y_subdf.columns:
                self.build_dataframe.update({i: y_subdf.loc[:, i]})
        array_creator.y_cols = y_subdf.columns
        instructions['parameters']['bool_trend'] = bool_trend
        instructions['parameters']['put'] = put_data.id
        return instructions

    def instructions_classification(self, _, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        self.task_type[put_data.id] = put_data.type
        self.encoding[put_data.id] = 'ohe' if options['one_hot_encoding'] else None

        if 'dataframe' in self.tags.values():
            transpose = self.user_parameters.get(
                list(self.tags.keys())[list(self.tags.values()).index('dataframe')]).transpose
            separator = self.user_parameters.get(
                list(self.tags.keys())[list(self.tags.values()).index('dataframe')]).separator
            if not any(self.y_cls):
                file_name = options['sources_paths'][0]
                if transpose:
                    tmp_df = pd.read_csv(os.path.join(self.file_folder, file_name), sep=separator).T
                    tmp_df.columns = tmp_df.iloc[0]
                    tmp_df.drop(tmp_df.index[[0]], inplace=True)
                    data = tmp_df.loc[:, options['cols_names'][0].split(' ')]
                else:
                    data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                       usecols=options['cols_names'],
                                       sep=separator)
                column = data[options['cols_names'][0]].to_list()

                if options['categorical']:
                    classes_names = []
                    for elem in column:
                        if elem not in classes_names:
                            classes_names.append(elem)
                    self.classes_names[put_data.id] = classes_names
                    self.num_classes[put_data.id] = len(classes_names)
                    for elem in column:
                        self.y_cls.append(classes_names.index(elem))
                else:
                    self.minvalue_y = min(column)
                    self.maxvalue_y = max(column)
                    if len(options['ranges'].split(' ')) == 1:
                        border = max(column) / int(options['ranges'])
                        self.classes_names[put_data.id] = np.linspace(
                            border, self.maxvalue_y, int(options['ranges'])).tolist()
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
                                           sep=separator,
                                           nrows=1).values
                else:
                    data = pd.read_csv(os.path.join(self.file_folder, options['sources_paths'][0]),
                                           sep=separator,
                                           usecols=[0]).values
                tmp = []
                for i in data:
                    tmp.append(i[0])
                self.classes_names[put_data.id] = sorted(list(set(tmp)))
                self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

        elif options['sources_paths'][0].endswith('.csv'):
            file_name = options['sources_paths'][0]
            data = pd.read_csv(os.path.join(self.file_folder, file_name),
                               usecols=options['cols_names'])
            column = data[options['cols_names'][0]].to_list()
            classes_names = []
            for elem in column:
                if elem not in classes_names:
                    classes_names.append(elem)
            self.classes_names[put_data.id] = classes_names
            self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

        else:
            self.classes_names[put_data.id] = sorted(options['sources_paths'])
            self.num_classes[put_data.id] = len(self.classes_names[put_data.id])

        instructions['parameters'] = {'num_classes': self.num_classes[put_data.id],
                                      'one_hot_encoding': options['one_hot_encoding'],
                                      'classes_names': self.classes_names[put_data.id]}
        # instructions['instructions'] = {f'{put_data.id}_classification': [
        #                                     self.classes_names[put_data.id][i] for i in self.y_cls]}
        self.build_dataframe[f'{put_data.id}_classification'] = [
                                            self.classes_names[put_data.id][i] for i in self.y_cls]
        return instructions

    def instructions_regression(self, number_list: list, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        options['put'] = put_data.id

        self.encoding[put_data.id] = None
        self.task_type[put_data.id] = put_data.type

        if options['scaler'] == 'min_max_scaler' or options['scaler'] == 'standard_scaler':
            if options['scaler'] == 'min_max_scaler':
                array_creator.scaler[put_data.id] = MinMaxScaler()
            if options['scaler'] == 'standard_scaler':
                array_creator.scaler[put_data.id] = StandardScaler()
            array_creator.scaler[put_data.id].fit(np.array(number_list).reshape(-1, 1))

        instructions['parameters'] = options
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_regression': number_list}
        else:
            self.build_dataframe[f'{put_data.id}_regression'] = number_list

        return instructions

    def instructions_segmentation(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData]):

        options = put_data.parameters.native()
        instructions: dict = {}
        shape: tuple = ()

        self.classes_names[put_data.id] = options['classes_names']
        self.classes_colors[put_data.id] = options['classes_colors']
        self.num_classes[put_data.id] = len(options['classes_names'])
        self.encoding[put_data.id] = 'ohe'
        self.task_type[put_data.id] = put_data.type

        for key, value in self.tags.items():
            if value == 'image':
                shape = (self.user_parameters.get(key).height, self.user_parameters.get(key).width)

        instructions['parameters'] = {'mask_range': options['mask_range'],
                                      'num_classes': len(options['classes_names']),
                                      'shape': shape,
                                      'classes_colors': options['classes_colors']
                                      }

        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_segmentation': paths_list}
        else:
            self.build_dataframe[f'{put_data.id}_segmentation'] = paths_list

        return instructions

    def instructions_object_detection(self, paths_list: list, put_data: Union[CreationInputData, CreationOutputData]):

        data = {}
        instructions = {}
        parameters = {}
        class_names = []

        for path in paths_list:
            if not path.endswith('.txt'):
                paths_list.remove(path)
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

        instructions['parameters'] = parameters
        self.build_dataframe[f'{put_data.id}_object_detection'] = paths_list

        return instructions

    def instructions_text_segmentation(self, _, put_data: Union[CreationInputData, CreationOutputData]):

        """
        Args:
            _
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

            # words = ' '.join(words)

            return words, indexes

        options = put_data.parameters.native()
        instructions: dict = {}
        text: list = []
        text_sliced: list = []
        text_segm: list = []
        text_segm_data: list = []
        text_segm_sliced: list = []
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        self.classes_names[put_data.id] = open_tags
        self.num_classes[put_data.id] = len(open_tags)
        self.encoding[put_data.id] = 'multi'
        self.peg = [0]

        for i, value in self.tags.items():
            if value == 'text':
                for key, txt_file in self.temporary[i].items():
                    if txt_file:
                        text_instr, segment_instr = get_samples(txt_file, open_tags, close_tags)
                        text.append(text_instr)
                        text_segm.append(segment_instr)
                length = self.user_parameters.get(i).dict()['length']
                step = self.user_parameters.get(i).dict()['step']
                text_mode = self.user_parameters.get(i).dict()['text_mode']
                max_words = self.user_parameters.get(i).dict()['max_words']

                for idx in range(len(text)):
                    if text_mode == TextModeChoice.completely:
                        text_sliced.append(' '.join(text[idx][0:max_words]))
                        text_segm_data.append(text_segm[idx][0:max_words])
                        text_segm_sliced.append([0, max_words])
                    elif text_mode == TextModeChoice.length_and_step:
                        max_length = len(text[idx])
                        cur_step = 0
                        stop_flag = False
                        while not stop_flag:
                            text_sliced.append(' '.join(text[idx][cur_step:cur_step + length]))
                            text_segm_data.append(text_segm[idx][cur_step:cur_step + length])
                            text_segm_sliced.append([cur_step, cur_step + length])
                            cur_step += step
                            if cur_step + length > max_length:
                                stop_flag = True

                self.build_dataframe[f'{i}_text'] = text_sliced
                self.build_dataframe[f'{i}_text_slice'] = text_segm_sliced
                self.build_dataframe[f'{put_data.id}_text_segmentation'] = text_segm_data
                self.build_dataframe[f'{put_data.id}_text_segmentation_slice'] = text_segm_sliced

            break

        self.peg.append(len(text_segm_sliced))
        instructions['parameters'] = {'num_classes': len(open_tags),
                                      'put': put_data.id}
        if options.get('deploy', bool):
            instructions['instructions'] = {f'{put_data.id}_text_segmentation': text_segm_data,
                                            f'{put_data.id}_text_segmentation_slice': text_segm_sliced}

        return instructions
