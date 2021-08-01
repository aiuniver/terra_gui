from terra_ai.data.datasets.extra import SourceModeChoice
from terra_ai.data.datasets.creation import SourceData

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import utils
from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from PIL import Image, ImageColor
from librosa import load as librosa_load
import librosa.feature as librosa_feature
import os
import random
import numpy as np
import pandas as pd
import pathlib
import re
import pymorphy2
import shutil
from gensim.models.word2vec import Word2Vec
from tqdm.notebook import tqdm
# from io import open as io_open
from terra_ai.guiexchange import Exchange
import joblib
import requests
from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone
import json
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2

tr2dj_obj = Exchange()

__version__ = 1.013


class CreateDTS(object):

    def __init__(self, trds_path='/content/drive/MyDrive/TerraAI/datasets',
                 exch_obj=tr2dj_obj):

        self.Exch = exch_obj
        self.django_flag: bool = False
        if self.Exch.property_of != 'TERRA':
            self.django_flag = True

        self.dataloader = Dataloader()
        self.createarray = None

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
        self.dataset: dict = {}

        self.y_cls: list = []
        self.sequence: list = []
        self.peg: list = []
        self.iter: int = 0
        self.mode: str = ''
        self.split_sequence: dict = {}
        self.temporary: dict = {}

        pass

    @staticmethod
    def _set_datatype(shape) -> str:

        datatype = {0: 'DIM',
                    1: 'DIM',
                    2: '1D',
                    3: '2D',
                    4: '3D',
                    5: '4D'
                    }

        return datatype[len(shape)]

    def load_data(self, strict_object):

        self.dataloader.load_data(strict_object=strict_object)

        self.zip_params = json.loads(strict_object.json())
        self.file_folder = self.dataloader.file_folder

        pass

    def create_dataset(self, dataset_dict: dict):

        self.createarray = CreateArray(file_folder=self.file_folder)

        self.name = dataset_dict['parameters']['name']
        self.divide_ratio = (dataset_dict['parameters']['train_part'], dataset_dict['parameters']['val_part'],
                             dataset_dict['parameters']['test_part'])
        self.source = 'custom dataset'
        self.user_tags = dataset_dict['parameters']['user_tags']
        self.use_generator = dataset_dict['parameters']['use_generator']

        for key in dataset_dict['inputs'].keys():
            self.tags[key] = dataset_dict['inputs'][key]['tag']
            self.input_names[key] = dataset_dict['inputs'][key]['name']
            self.user_parameters[key] = dataset_dict['inputs'][key]['parameters']
        for key in dataset_dict['outputs'].keys():
            self.tags[key] = dataset_dict['outputs'][key]['tag']
            self.output_names[key] = dataset_dict['outputs'][key]['name']
            self.user_parameters[key] = dataset_dict['outputs'][key]['parameters']

        # Создаем входные инструкции
        self.iter = 0
        self.mode = 'input'
        for inp in dataset_dict['inputs']:
            self.iter += 1
            self.instructions['inputs'][f'{self.mode}_{self.iter}'] = getattr(self,
                                                                              f"instructions_{self.tags[f'{self.mode}_{self.iter}']}")(
                **dataset_dict['inputs'][f'{self.mode}_{self.iter}']['parameters'])
        # Создаем выходные инструкции
        self.iter = 0
        self.mode = 'output'
        for out in dataset_dict['outputs']:
            self.iter += 1
            self.instructions['outputs'][out] = getattr(self, f"instructions_{self.tags[out]}")(
                **dataset_dict['outputs'][out]['parameters'])

        # Получаем входные параметры
        for key in self.instructions['inputs'].keys():
            array = getattr(self.createarray, f'create_{self.tags[key]}')(
                self.instructions['inputs'][key]['instructions'][0], **self.instructions['inputs'][key]['parameters'])
            self.input_shape[key] = array.shape
            self.input_dtype[key] = str(array.dtype)
            self.input_datatype[key] = self._set_datatype(array.shape)
        # Получаем выходные параметры
        for key in self.instructions['outputs'].keys():
            array = getattr(self.createarray, f'create_{self.tags[key]}')(
                self.instructions['outputs'][key]['instructions'][0], **self.instructions['outputs'][key]['parameters'])
            if isinstance(array, tuple):
                for i in range(len(array)):
                    self.output_shape[key.replace(key[-1], str(int(key[-1]) + i))] = array[i].shape
                    self.output_dtype[key.replace(key[-1], str(int(key[-1]) + i))] = str(array[i].dtype)
                    self.output_datatype[key.replace(key[-1], str(int(key[-1]) + i))] =\
                        self._set_datatype(array[i].shape)
            else:
                self.output_shape[key] = array.shape
                self.output_dtype[key] = str(array.dtype)
                self.output_datatype[key] = self._set_datatype(array.shape)

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
            if 'text' in self.tags.keys():  # if 'txt_list' in self.createarray.__dict__.keys():
                with open(os.path.join(self.trds_path, f'dataset {self.name}', 'instructions', 'txt_list.json'),
                          'w') as fp:
                    json.dump(self.createarray.txt_list, fp)
        else:
            # Сохранение датасета с NumPy
            for key in self.instructions['inputs'].keys():
                x: list = []
                for i in range(self.limit):
                    x.append(getattr(self.createarray, f"create_{self.tags[key]}")(
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
                    y_4: list = []
                    y_5: list = []
                    y_6: list = []
                    for i in range(self.limit):
                        arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                            self.instructions['outputs'][key]['instructions'][i],
                            **self.instructions['outputs'][key]['parameters'])
                        y_1.append(arrays[0])
                        y_2.append(arrays[1])
                        y_3.append(arrays[2])
                        y_4.append(arrays[3])
                        y_5.append(arrays[4])
                        y_6.append(arrays[5])

                    splits = ['train', 'val', 'test']
                    for spl_seq in splits:
                        for i in range(len(splits)):
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1])))] = np.array(y_1)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1]) + 1))] = np.array(y_2)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1]) + 2))] = np.array(y_3)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1]) + 3))] = np.array(y_4)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1]) + 4))] = np.array(y_5)[
                                self.split_sequence[spl_seq]]
                            self.Y[spl_seq][key.replace(key[-1], str(int(key[-1]) + 5))] = np.array(y_6)[
                                self.split_sequence[spl_seq]]
                else:
                    y: list = []
                    for i in range(self.limit):
                        y.append(getattr(self.createarray, f"create_{self.tags[key]}")(
                            self.instructions['outputs'][key]['instructions'][i],
                            **self.instructions['outputs'][key]['parameters']))
                    self.Y['train'][key] = np.array(y)[self.split_sequence['train']]
                    self.Y['val'][key] = np.array(y)[self.split_sequence['val']]
                    self.Y['test'][key] = np.array(y)[self.split_sequence['test']]

            for sample in self.X.keys():
                # os.makedirs(os.path.join(self.trds_path, 'arrays', sample), exist_ok=True)
                for inp in self.X[sample].keys():
                    os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample), exist_ok=True)
                    joblib.dump(self.X[sample][inp],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample, f'{inp}.gz'))

            for sample in self.Y.keys():
                for inp in self.Y[sample].keys():
                    os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample), exist_ok=True)
                    joblib.dump(self.Y[sample][inp],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', sample, f'{inp}.gz'))

        if self.createarray.scaler:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'scalers'), exist_ok=True)
        if self.createarray.tokenizer:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer'), exist_ok=True)
        if self.createarray.word2vec:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec'), exist_ok=True)
        if self.createarray.augmentation:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'augmentation'), exist_ok=True)
        # if self.createarray.tsgenerator:
        #     os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tsgenerator'), exist_ok=True)

        for scaler in self.createarray.scaler.keys():
            if self.createarray.scaler[scaler]:
                joblib.dump(self.createarray.scaler[scaler],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'scalers', f'{scaler}.gz'))
        for tok in self.createarray.tokenizer.keys():
            if self.createarray.tokenizer[tok]:
                joblib.dump(self.createarray.tokenizer[tok],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer', f'{tok}.gz'))
        for w2v in self.createarray.word2vec.keys():
            if self.createarray.word2vec[w2v]:
                joblib.dump(self.createarray.word2vec[w2v],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec', f'{w2v}.gz'))
        for aug in self.createarray.augmentation.keys():
            if self.createarray.augmentation[aug]:
                joblib.dump(self.createarray.augmentation[aug],
                            os.path.join(self.trds_path, f'dataset {self.name}', 'augmentation', f'{aug}.gz'))
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
        if 'augmentation' in options.keys():
            aug_parameters = []
            for key, value in options['augmentation'].items():
                aug_parameters.append(getattr(iaa, key)(**value))
            self.createarray.augmentation[f'{self.mode}_{self.iter}'] = iaa.Sequential(aug_parameters,
                                                                                       random_order=True)
            del options['augmentation']

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_video(self, **options):

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

        def read_text(file_path, lower, filters, split) -> str:

            # del_symbols = ['\n', '\t', '\ufeff']
            # if options['delete_symbols']:
            #     del_symbols += options['delete_symbols'].split(' ')
            #
            # with io_open(file_path, encoding='utf-8', errors='ignore') as f:
            #     text = f.read()
            #     for del_symbol in del_symbols:
            #         text = text.replace(del_symbol, ' ')
            # for put, tag in self.tags.items():
            #     if tag == 'text_segmentation':
            #         open_symbol = self.user_parameters[put]['open_tags'].split(' ')[0][0]
            #         close_symbol = self.user_parameters[put]['open_tags'].split(' ')[0][-1]
            #         text = re.sub(open_symbol, f" {open_symbol}", text)
            #         text = re.sub(close_symbol, f"{close_symbol} ", text)
            #         break
            with open(os.path.join(self.file_folder, file_path), 'r') as txt:
                text = txt.read()
            text = ' '.join(text_to_word_sequence(text, **{'lower': lower, 'filters': filters, 'split': split}))

            return text

        def apply_pymorphy(text, morphy) -> str:

            words_list = text.split(' ')
            words_list = [morphy.parse(w)[0].normal_form for w in words_list]

            return ' '.join(words_list)

        txt_list: dict = {}
        lower: bool = True
        filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
        split: str = ' '

        for key, value in self.tags.items():
            if value == 'text_segmentation':
                open_tags = self.user_parameters[key]['open_tags']
                close_tags = self.user_parameters[key]['close_tags']
                tags = f'{open_tags} {close_tags}'
                for ch in filters:
                    if ch in set(tags):
                        filters = filters.replace(ch, '')
                break

        if options['file_info']['path_type'] == 'path_folder':
            for folder_name in options['file_info']['path']:
                for directory, folder, file_name in sorted(os.walk(os.path.join(self.file_folder, folder_name))):
                    if file_name:
                        file_folder = directory.replace(self.file_folder, '')[1:]
                        for name in sorted(file_name):
                            file_path = os.path.join(file_folder, name)
                            txt_list[file_path] = read_text(file_path, lower, filters, split)
        elif options['file_info']['path_type'] == 'path_file':
            for file_name in options['file_info']['path']:
                data = pd.read_csv(os.path.join(self.file_folder, file_name),
                                   usecols=options['file_info']['cols_name'])
                column = data[options['file_info']['cols_name'][0]].to_list()
                for idx, elem in column:
                    txt_list[str(idx)] = elem

        if options['pymorphy']:
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_list.items():
                txt_list[key] = apply_pymorphy(value, pymorphy)

        # self.createarray.txt_list[f'{self.mode}_{self.iter}'] = {}
        # for key, value in txt_list.items():
        #     self.createarray.txt_list[f'{self.mode}_{self.iter}'][key] = \
        #         self.createarray.tokenizer[f'{self.mode}_{self.iter}'].texts_to_sequences([value])[0]

        if options['word_to_vec']:
            txt_list_w2v = []
            for elem in list(txt_list.values()):
                txt_list_w2v.append(elem.split(' '))
            self.createarray.create_word2vec(self.mode, self.iter, txt_list_w2v, **{'size': options['word_to_vec_size'],
                                                                                    'window': 10,
                                                                                    'min_count': 1,
                                                                                    'workers': 10,
                                                                                    'iter': 10})
        else:
            self.createarray.create_tokenizer(self.mode, self.iter, **{'num_words': options['max_words_count'],
                                                                       'filters': filters,
                                                                       'lower': lower,
                                                                       'split': split,
                                                                       'char_level': False,
                                                                       'oov_token': '<UNK>'})
            self.createarray.tokenizer[f'{self.mode}_{self.iter}'].fit_on_texts(list(txt_list.values()))

        # if 'text_segmentation' not in self.tags.values():
        #     y_cls = []
        #     cls_idx = 0
        #     length = options['x_len']
        #     stride = options['step']
        #     peg_idx = 0
        #     self.peg.append(0)
        #     for key in sorted(self.createarray.txt_list[f'{self.mode}_{self.iter}'].keys()):
        #         index = 0
        #         while index + length <= len(self.createarray.txt_list[f'{self.mode}_{self.iter}'][key]):
        #             instr.append({'file': key, 'slice': [index, index + length]})
        #             peg_idx += 1
        #             index += stride
        #             y_cls.append(cls_idx)
        #         self.peg.append(peg_idx)
        #         cls_idx += 1

        instr: list = []
        y_cls: list = []
        peg_idx: int = 0
        cls_idx: int = 0
        prev_class: str = sorted(txt_list.keys())[0].split('/')[-2]
        self.peg.append(0)

        for key, value in sorted(txt_list.items()):
            cur_class = key.split('/')[-2]
            if options['txt_mode'] == 'Целиком':
                instr.append({key: [0, options['max_words']]})
                if cur_class != prev_class:
                    cls_idx += 1
                    self.peg.append(peg_idx)
                    prev_class = cur_class
                peg_idx += 1
                y_cls.append(cls_idx)
            elif options['txt_mode'] == 'По длине и шагу':
                max_length = len(value.split(' '))
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    instr.append({key: [cur_step, cur_step + options['length']]})
                    cur_step += options['step']
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
        self.createarray.txt_list[f'{self.mode}_{self.iter}'] = txt_list

        instructions = {'instructions': instr,
                        'parameters': {'embedding': options['embedding'],
                                       'bag_of_words': options['bag_of_words'],
                                       'word_to_vec': options['word_to_vec'],
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
        self.createarray.df_with_y = pd.read_csv(
            os.path.join(self.file_folder, options['file_info']['path'][0]), usecols=(str_to_list(
                options['file_info']['cols_name'][0], general_df.columns) + str_to_list(options['y_col'],
                                                                                        general_df.columns)))
        self.createarray.df_with_y.sort_values(by=options['y_col'], inplace=True, ignore_index=True)

        self.peg.append(0)
        for i in range(len(self.createarray.df_with_y.loc[:, options['y_col']]) - 1):
            if self.createarray.df_with_y.loc[:, options['y_col']][i] != \
                    self.createarray.df_with_y.loc[:, options['y_col']][i + 1]:
                self.peg.append(i + 1)
        self.peg.append(len(self.createarray.df_with_y))

        self.createarray.df = self.createarray.df_with_y.iloc[:, str_to_list(
            options['file_info']['cols_name'][0], self.createarray.df_with_y.columns)]

        instructions = {'instructions': np.arange(0, len(self.createarray.df)).tolist(),
                        'parameters': {'put': f'{self.mode}_{self.iter}'}}

        if 'MinMaxScaler' or 'StandardScaler' in options.keys():
            self.createarray.scaler[f'{self.mode}_{self.iter}'] = {}
            if 'MinMaxScaler' in options.keys():
                instructions['parameters']['MinMaxScaler'] = str_to_list(str_numbers=options['MinMaxScaler'],
                                                                         df_cols=self.createarray.df.columns)
                self.createarray.scaler[f'{self.mode}_{self.iter}']['MinMaxScaler'] = MinMaxScaler()
                self.createarray.scaler[f'{self.mode}_{self.iter}']['MinMaxScaler'].fit(
                    self.createarray.df.iloc[:, instructions['parameters']['MinMaxScaler']].to_numpy().reshape(-1, 1))

            if 'StandardScaler' in options.keys():
                instructions['parameters']['StandardScaler'] = str_to_list(options['StandardScaler'],
                                                                           self.createarray.df.columns)
                self.createarray.scaler[f'{self.mode}_{self.iter}']['StandardScaler'] = StandardScaler()
                self.createarray.scaler[f'{self.mode}_{self.iter}']['StandardScaler'].fit(
                    self.createarray.df.iloc[:, instructions['parameters']['StandardScaler']].to_numpy().reshape(-1, 1))

        if 'Categorical' in options.keys():
            instructions['parameters']['Categorical'] = {}
            instructions['parameters']['Categorical']['lst_cols'] = str_to_list(options['Categorical'],
                                                                                self.createarray.df.columns)
            for i in instructions['parameters']['Categorical']['lst_cols']:
                instructions['parameters']['Categorical'][f'col_{i}'] = np.unique(
                    self.createarray.df.iloc[:, i]).tolist()

        if 'Categorical_ranges' in options.keys():
            instructions['parameters']['Categorical_ranges'] = {}
            instructions['parameters']['Categorical_ranges']['lst_cols'] = str_to_list(
                options['Categorical_ranges']['cols'], self.createarray.df.columns)
            for i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                instructions['parameters']['Categorical_ranges'][f'col_{i}'] = {}
                for j in range(len(options['Categorical_ranges'][f'col_{i + 1}'].split(' '))):
                    instructions['parameters']['Categorical_ranges'][f'col_{i}'][f'range_{j}'] = int(
                        options['Categorical_ranges'][f'col_{i + 1}'].split(' ')[j])

        if 'one_hot_encoding' in options.keys():
            instructions['parameters']['one_hot_encoding'] = {}
            instructions['parameters']['one_hot_encoding']['lst_cols'] = str_to_list(options['one_hot_encoding'],
                                                                                     self.createarray.df.columns)
            for i in instructions['parameters']['one_hot_encoding']['lst_cols']:
                if i in instructions['parameters']['Categorical_ranges']['lst_cols']:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        options['Categorical_ranges'][f'col_{i + 1}'].split(' '))
                else:
                    instructions['parameters']['one_hot_encoding'][f'col_{i}'] = len(
                        np.unique(self.createarray.df.iloc[:, i]))

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
                self.createarray.scaler[f'{self.mode}_{self.iter}'] = MinMaxScaler()
            if options['scaler'] == 'StandardScaler':
                self.createarray.scaler[f'{self.mode}_{self.iter}'] = StandardScaler()
            self.createarray.scaler[f'{self.mode}_{self.iter}'].fit(np.array(instr).reshape(-1, 1))

        instructions['instructions'] = instr
        instructions['parameters'] = options

        return instructions

    def instructions_segmentation(self, **options):

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

        # for key, value in self.tags.items():
        #     if value == 'images':
                # parameters['height'] = self.user_parameters[key]['height']
                # parameters['width'] = self.user_parameters[key]['width']
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
        self.createarray.txt_list[f'{self.mode}_{self.iter}'] = {}

        for key, value in self.tags.items():
            if value == 'text':
                tags_indexes = np.array([self.createarray.tokenizer[key].word_index[idx] for idx in tags])
                for txt_file in self.createarray.txt_list[key].keys():
                    text_instr, segment_instr = get_ohe_samples(self.createarray.txt_list[key][txt_file], tags_indexes)
                    self.createarray.txt_list[f'{self.mode}_{self.iter}'][txt_file] = segment_instr
                    self.createarray.txt_list[key][txt_file] = text_instr

                length = self.user_parameters[key]['x_len']
                stride = self.user_parameters[key]['step']
                peg_idx = 0
                self.peg = []
                self.peg.append(0)
                for path in sorted(self.createarray.txt_list[f'{self.mode}_{self.iter}'].keys()):
                    index = 0
                    while index + length <= len(self.createarray.txt_list[key][path]):
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


class Dataloader(object):

    def __init__(self, path=mkdtemp(), trds_path='/content/drive/MyDrive/TerraAI/datasets'):

        self.file_folder: str = ''
        self.save_path = path
        self.trds_path = trds_path
        self.django_flag = False

        pass

    def _get_zipfiles(self) -> list:

        return os.listdir(os.path.join(self.trds_path, 'sources'))

    @staticmethod
    def unzip(file_folder: str, zip_name: str):

        file_path = pathlib.Path(os.path.join(file_folder, 'tmp', zip_name))
        temp_folder = os.path.join(file_folder, 'tmp')
        os.makedirs(temp_folder, exist_ok=True)
        shutil.unpack_archive(file_path, file_folder)
        shutil.rmtree(temp_folder, ignore_errors=True)

        pass

    @staticmethod
    def download(link: str, file_folder: str, file_name: str):

        resp = requests.get(link, stream=True)
        total = int(resp.headers.get('content-length', 0))
        idx = 0
        with open(os.path.join(file_folder, 'tmp', file_name), 'wb') as out_file, tqdm(
                desc=f"Загрузка архива {file_name}", total=total, unit='iB', unit_scale=True,
                unit_divisor=1024) as progress_bar:
            for data in resp.iter_content(chunk_size=1024):
                size = out_file.write(data)
                progress_bar.update(size)
                idx += size
                # if self.django_flag:
                #     if idx % 143360 == 0 or idx == progress_bar.total:
                #         progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                #                            f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                #         if idx == progress_bar.total:
                #             self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                #         else:
                #             self.Exch.print_progress_bar(progress_bar_status)

        pass

    def load_data(self, strict_object):

        if strict_object.mode == SourceModeChoice.Terra:
            self.load_from_terra(strict_object.value)
        elif strict_object.mode == SourceModeChoice.URL:
            self.load_from_url(strict_object.value)
        elif strict_object.mode == SourceModeChoice.GoogleDrive:
            self.load_from_google(strict_object.value)

        pass

    def load_from_terra(self, name: str):

        file_folder = None
        data = {
            'трейдинг': ['trading.zip'],
            'автомобили': ['cars.zip'],
            'умный_дом': ['smart_home.zip'],
            'квартиры': ['flats.zip'],
            # 'диалоги': ['dialog.txt'],
            'автомобили_3': ['cars_3.zip'],
            'заболевания': ['diseases.zip'],
            'договоры': ['docs.zip'],
            'самолеты': ['planes.zip'],
            # 'болезни': ['origin.zip', 'segmentation.zip'],
            'губы': ['lips.zip'],
            # 'жанры_музыки': ['genres.zip'],
            'sber': ['sber.zip']
        }

        for file_name in data[name]:
            file_folder = pathlib.Path(self.save_path).joinpath(name)
            os.makedirs(file_folder, exist_ok=True)
            os.makedirs(os.path.join(file_folder, 'tmp'), exist_ok=True)
            link = 'https://storage.googleapis.com/terra_ai/DataSets/Numpy/' + file_name
            self.download(link, file_folder, file_name)
            if 'zip' in file_name:
                self.unzip(file_folder, file_name)
        self.file_folder = str(file_folder)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        pass

    def load_from_url(self, link: str):

        file_name = link.split('/')[-1]
        file_folder = pathlib.Path(os.path.join(self.save_path, file_name))
        if '.' in file_name:
            name = file_name[:file_name.rfind('.')]
            file_folder = pathlib.Path(os.path.join(self.save_path, name))
        os.makedirs(file_folder, exist_ok=True)
        os.makedirs(os.path.join(file_folder, 'tmp'), exist_ok=True)
        self.download(link, file_folder, file_name)
        if 'zip' in file_name or 'zip' in link:
            self.unzip(file_folder, file_name)
        self.file_folder = str(file_folder)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        pass

    def load_from_google(self, filepath: str):

        zip_name = str(filepath).split('/')[-1]
        name = zip_name[:zip_name.rfind('.')]
        file_folder = os.path.join(self.save_path, name)
        shutil.unpack_archive(filepath, file_folder)
        self.file_folder = str(file_folder)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        pass


class CreateArray(object):

    def __init__(self, **options):

        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.augmentation: dict = {}
        self.temporary: dict = {'bounding_boxes': {}}
        self.df = None

        self.file_folder = None
        self.txt_list: dict = {}

        for key, value in options.items():
            self.__dict__[key] = value

    @staticmethod
    def yolo_to_imgaug(args, shape):

        height, width = shape

        class_num = int(args[0])
        x_pos = float(args[1])
        y_pos = float(args[2])
        x_size = float(args[3])
        y_size = float(args[4])

        x1 = x_pos * width - (x_size * width / 2)
        y1 = y_pos * height - (y_size * height / 2)
        x2 = x_size * width + x1
        y2 = y_size * height + y1

        return [class_num, x1, y1, x2, y2]

    @staticmethod
    def imgaug_to_yolo(args, shape=(416, 416)):

        height, width = shape

        class_num = int(args[0])
        x1 = float(args[1])
        y1 = float(args[2])
        x2 = float(args[3])
        y2 = float(args[4])

        x_pos = x1 / width + ((x2 - x1) / width / 2)
        y_pos = y1 / height + ((y2 - y1) / height / 2)
        x_size = (x2 - x1) / width
        y_size = (y2 - y1) / height

        return_args = [class_num, x_pos, y_pos, x_size, y_size]

        for r in return_args[1:]:
            if r > 1:
                return ()
            if r < 0:
                return ()

        return return_args

    def create_images(self, image_path: str, **options):

        shape = (options['height'], options['width'])
        # img = cv2.imread(os.path.join(self.file_folder, image_path)).reshape(*shape, 3)
        img = load_img(os.path.join(self.file_folder, image_path), target_size=shape)
        array = img_to_array(img, dtype=np.uint8)
        if options['net'] == 'Linear':
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['put'] in self.augmentation.keys():
            if 'object_detection' in options.keys():
                txt_path = image_path[:image_path.rfind('.')] + '.txt'
                with open(os.path.join(self.file_folder, txt_path), 'r') as b_boxes:
                    bounding_boxes = b_boxes.read()

                current_boxes = []
                for elem in bounding_boxes.split('\n'):
                    # b_box = self.yolo_to_imgaug(elem.split(' '), shape=array.shape[:2])
                    if elem:
                        b_box = elem.split(',')
                        b_box = [int(x) for x in b_box]
                        current_boxes.append(
                            BoundingBox(
                                **{'label': b_box[4], 'x1': b_box[0], 'y1': b_box[1], 'x2': b_box[2], 'y2': b_box[3]}))

                bbs = BoundingBoxesOnImage(current_boxes, shape=array.shape)
                array, bbs_aug = self.augmentation[options['put']](image=array, bounding_boxes=bbs)
                list_of_bounding_boxes = []
                for elem in bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes:
                    bb = elem.__dict__
                    # b_box_coord = self.imgaug_to_yolo([bb['label'], bb['x1'], bb['y1'], bb['x2'], bb['y2']],
                    #                                   shape=array.shape[:2])
                    # if b_box_coord != ():
                    if bb:
                        list_of_bounding_boxes.append([bb['x1'], bb['y1'], bb['x2'], bb['y2'], bb['label']])

                self.temporary['bounding_boxes'][txt_path] = list_of_bounding_boxes
            else:
                array = self.augmentation[options['put']](image=array)

        array = array / 255

        return array.astype('float32')

    def create_video(self, video_path, **options) -> np.ndarray:

        """

        Args:
            video_path: dict
                Путь к файлу: [начало, конец]
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
            array: np.ndarray
                Массив видео.

        """

        def resize_frame(one_frame, original_shape, target_shape, frame_mode):

            resized = None

            if frame_mode == 'Растянуть':
                resized = resize_layer(one_frame[None, ...])
                resized = resized.numpy().squeeze().astype('uint8')
            elif frame_mode == 'Сохранить пропорции':
                # height
                resized = one_frame.copy()
                if original_shape[0] > target_shape[0]:
                    resized = resized[int(original_shape[0] / 2 - target_shape[0] / 2):int(
                        original_shape[0] / 2 - target_shape[0] / 2) + target_shape[0], :]
                else:
                    black_bar = np.zeros((int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    resized = np.concatenate((resized, black_bar))
                # width
                if original_shape[1] > target_shape[1]:
                    resized = resized[:, int(original_shape[1] / 2 - target_shape[1] / 2):int(
                        original_shape[1] / 2 - target_shape[1] / 2) + target_shape[1]]
                else:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    resized = np.concatenate((resized, black_bar), axis=1)

            return resized

        def add_frames(video_array, fill_mode, frames_to_add, total_frames):

            frames: np.ndarray = np.array([])

            if fill_mode == 'Черными кадрами':
                frames = np.zeros((frames_to_add, *shape, 3), dtype='uint8')
            elif fill_mode == 'Средним значением':
                mean = np.mean(video_array, axis=0, dtype='uint16')
                frames = np.full((frames_to_add, *mean.shape), mean, dtype='uint8')
            elif fill_mode == 'Последними кадрами':
                # cur_frames = video_array.shape[0]
                if total_frames > frames_to_add:
                    frames = np.flip(video_array[-frames_to_add:], axis=0)
                elif total_frames <= frames_to_add:
                    for i in range(frames_to_add // total_frames):
                        frames = np.flip(video_array[-total_frames:], axis=0)
                        video_array = np.concatenate((video_array, frames), axis=0)
                    if frames_to_add + total_frames != video_array.shape[0]:
                        frames = np.flip(video_array[-(frames_to_add + total_frames - video_array.shape[0]):], axis=0)
            video_array = np.concatenate((video_array, frames), axis=0)

            return video_array

        array = []
        shape = (options['height'], options['width'])
        [[file_name, video_range]] = video_path.items()
        frames_count = video_range[1] - video_range[0]
        resize_layer = Resizing(*shape)

        cap = cv2.VideoCapture(os.path.join(self.file_folder, file_name))
        width = int(cap.get(3))
        height = int(cap.get(4))
        max_frames = int(cap.get(7))
        cap.set(1, video_range[0])
        try:
            for _ in range(frames_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if shape != (height, width):
                    frame = resize_frame(one_frame=frame,
                                         original_shape=(height, width),
                                         target_shape=shape,
                                         frame_mode=options['frame_mode'])
                frame = frame[:, :, [2, 1, 0]]
                array.append(frame)
        finally:
            cap.release()

        array = np.array(array)
        if max_frames < frames_count:
            array = add_frames(video_array=array,
                               fill_mode=options['fill_mode'],
                               frames_to_add=frames_count - max_frames,
                               total_frames=max_frames)

        return array

    def create_text(self, sample: dict, **options):

        """

        Args:
            sample: dict
                - file: Название файла.
                - slice: Индексы рассматриваемой части последовательности
            **options: Параметры обработки текста:
                embedding: Tokenizer object, bool
                    Перевод в числовую последовательность.
                bag_of_words: Tokenizer object, bool
                    Перевод в формат bag_of_words.
                word_to_vec: Word2Vec object, bool
                    Перевод в векторное представление Word2Vec.
                put: str
                    Индекс входа или выхода.

        Returns:
            array: np.ndarray
                Массив текстового вектора.
        """

        array = []
        [[filepath, slicing]] = sample.items()
        text = self.txt_list[options['put']][filepath].split(' ')[slicing[0]:slicing[1]]

        if options['embedding']:
            array = self.tokenizer[options['put']].texts_to_sequences([text])[0]
        elif options['bag_of_words']:
            array = self.tokenizer[options['put']].texts_to_matrix([text])[0]
        elif options['word_to_vec']:
            for word in text:
                array.append(self.word2vec[options['put']][word])

        if len(array) < slicing[1] - slicing[0]:
            words_to_add = [1 for _ in range((slicing[1] - slicing[0]) - len(array))]
            array += words_to_add

        # for key, value in options.items():
        #     if value:
        #         if key == 'bag_of_words':
        #             array = self.tokenizer[options['put']].sequences_to_matrix([array]).astype('uint16')
        #         elif key == 'word_to_vec':
        # reverse_tok = {}
        # words_list = []
        # for word, index in self.tokenizer[options['put']].word_index.items():
        #     reverse_tok[index] = word
        # for idx in array:
        #     words_list.append(reverse_tok[idx])
        #     array = []
        #     for word in words_list:
        #         array.append(self.word2vec[options['put']].wv[word])
        # break

        array = np.array(array)

        return array

    def create_audio(self):

        pass

    def create_dataframe(self, row_number: int, **options):
        """
            Args:
                row_number: номер строки с сырыми данными датафрейма,
                **options: Параметры обработки колонок:
                    MinMaxScaler: лист индексов колонок для обработки
                    StandardScaler: лист индексов колонок для обработки
                    Categorical: лист индексов колонок для перевода по готовым категориям
                    Categorical_ranges: лист индексов колонок для перевода по категориям по диапазонам
                    one_hot_encoding: лист индексов колонок для перевода в ОНЕ
                    put: str  Индекс входа или выхода.
            Returns:
                array: np.ndarray
                    Массив вектора обработанных данных.
        """
        row = self.df.loc[row_number].copy().tolist()

        if 'StandardScaler' in options.keys():
            for i in options['StandardScaler']:
                row[i] = self.scaler[options['put']]['StandardScaler'].transform(
                    np.array([row[i]]).reshape(-1, 1)).tolist()

        if 'MinMaxScaler' in options.keys():
            for i in options['MinMaxScaler']:
                row[i] = self.scaler[options['put']]['MinMaxScaler'].transform(
                    np.array(row[i]).reshape(-1, 1)).tolist()

        if 'Categorical' in options.keys():
            for i in options['Categorical']['lst_cols']:
                row[i] = list(options['Categorical'][f'col_{i}']).index(row[i])

        if 'Categorical_ranges' in options.keys():
            for i in options['Categorical_ranges']['lst_cols']:
                for j in range(len(options['Categorical_ranges'][f'col_{i}'])):
                    if row[i] <= options['Categorical_ranges'][f'col_{i}'][f'range_{j}']:
                        row[i] = j
                        break

        if 'one_hot_encoding' in options.keys():
            for i in options['one_hot_encoding']['lst_cols']:
                row[i] = utils.to_categorical(row[i], options['one_hot_encoding'][f'col_{i}'], dtype='uint8').tolist()

        array = []
        for i in row:
            if type(i) == list:
                if type(i[0]) == list:
                    array.extend(i[0])
                else:
                    array.extend(i)
            else:
                array.append(i)

        array = np.array(array)

        return array

    def create_classification(self, index, **options):

        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        array = np.array(index)

        return array

    def create_regression(self, index, **options):

        if 'scaler' in options.keys():
            index = self.scaler[options['put']].transform(np.array(index).reshape(-1, 1)).reshape(1, )[0]
        array = np.array(index)

        return array

    def create_segmentation(self, image_path: str, **options: dict) -> np.ndarray:

        """

        Args:
            image_path: str
                Путь к файлу
            **options: Параметры сегментации:
                mask_range: int
                    Диапазон для каждого из RGB каналов.
                num_classes: int
                    Общее количество классов.
                shape: tuple
                    Размер картинки (высота, ширина).
                classes_colors: list
                    Список цветов для каждого класса.

        Returns:
            array: np.ndarray
                Массив принадлежности каждого пикселя к определенному классу в формате One-Hot Encoding.

        """

        def cluster_to_ohe(mask_image):

            mask_image = mask_image.reshape(-1, 3)
            km = KMeans(n_clusters=options['num_classes'])
            km.fit(mask_image)
            labels = km.labels_
            cl_cent = km.cluster_centers_.astype('uint8')[:max(labels) + 1]
            cl_mask = utils.to_categorical(labels, max(labels) + 1, dtype='uint8')
            cl_mask = cl_mask.reshape(options['shape'][0], options['shape'][1], cl_mask.shape[-1])

            mask_ohe = np.zeros(options['shape'])
            for k, rgb in enumerate(options['classes_colors']):
                mask = np.zeros(options['shape'])

                for j, cl_rgb in enumerate(cl_cent):
                    if rgb[0] in range(cl_rgb[0] - options['mask_range'], cl_rgb[0] + options['mask_range']) and \
                            rgb[1] in range(cl_rgb[1] - options['mask_range'], cl_rgb[1] + options['mask_range']) and \
                            rgb[2] in range(cl_rgb[2] - options['mask_range'], cl_rgb[2] + options['mask_range']):
                        mask = cl_mask[:, :, j]

                if k == 0:
                    mask_ohe = mask
                else:
                    mask_ohe = np.dstack((mask_ohe, mask))

            return mask_ohe

        img = load_img(path=os.path.join(self.file_folder, image_path), target_size=options['shape'])
        array = img_to_array(img, dtype=np.uint8)
        array = cluster_to_ohe(array)

        return array

    def create_text_segmentation(self, sample: dict, **options):

        array = []

        for elem in self.txt_list[options['put']][sample['file']][sample['slice'][0]:sample['slice'][1]]:
            tags = [0 for _ in range(options['num_classes'])]
            if elem:
                for idx in elem:
                    tags[idx] = 1
            array.append(tags)
        array = np.array(array, dtype='uint8')

        return array

    def create_timeseries(self):

        pass

    def create_object_detection(self, txt_path: str, **options):

        """

        Args:
            txt_path: str
                Путь к файлу
            **options: Параметры сегментации:
                height: int ######!!!!!!
                    Высота изображения.
                width: int ######!!!!!!
                    Ширина изображения.
                num_classes: int
                    Количество классов.

        Returns:
            array: np.ndarray
                Массивы в трёх выходах.

        """

        def bbox_iou(boxes1, boxes2):

            boxes1_area = boxes1[..., 2] * boxes1[..., 3]
            boxes2_area = boxes2[..., 2] * boxes2[..., 3]

            boxes1 = tf_concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
            boxes2 = tf_concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

            left_up = tf_maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = tf_minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = tf_maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area

            return 1.0 * inter_area / union_area

        # height: int = options['height']
        # width: int = options['width']
        num_classes: int = options['num_classes']
        zero_boxes_flag: bool = False
        strides = np.array([8, 16, 32])
        output_levels = len(strides)
        train_input_sizes = 416
        anchor_per_scale = 3
        yolo_anchors = [[[12, 16], [19, 36], [40, 28]],
                        [[36, 75], [76, 55], [72, 146]],
                        [[142, 110], [192, 243], [459, 401]]]
        anchors = (np.array(yolo_anchors).T / strides).T
        max_bbox_per_scale = 100
        train_input_size = random.choice([train_input_sizes])
        train_output_sizes = train_input_size // strides

        if self.temporary['bounding_boxes']:
            real_boxes = self.temporary['bounding_boxes'][txt_path]
        else:
            with open(os.path.join(self.file_folder, txt_path), 'r') as txt:
                bb_file = txt.read()
            real_boxes = []
            for elem in bb_file.split('\n'):
                tmp = []
                if elem:
                    for num in elem.split(','):
                        tmp.append(int(num))
                    real_boxes.append(tmp)

        if not real_boxes:
            zero_boxes_flag = True
            real_boxes = [[0, 0, 0, 0, 0]]
        real_boxes = np.array(real_boxes)
        label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale,
                           5 + num_classes)) for i in range(output_levels)]
        bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(output_levels)]
        bbox_count = np.zeros((output_levels,))

        for bbox in real_boxes:
            bbox_class_ind = int(bbox[4])
            bbox_coordinate = np.array(bbox[:4])
            one_hot = np.zeros(num_classes, dtype=np.float)
            one_hot[bbox_class_ind] = 0.0 if zero_boxes_flag else 1.0
            uniform_distribution = np.full(num_classes, 1.0 / num_classes)
            deta = 0.01
            smooth_one_hot = one_hot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                                        bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(output_levels):  # range(3):
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 0.0 if zero_boxes_flag else 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_one_hot

                    bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchor_per_scale)
                best_anchor = int(best_anchor_ind % anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 0.0 if zero_boxes_flag else 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return np.array(label_sbbox, dtype='float32'), np.array(sbboxes, dtype='float32'),\
               np.array(label_mbbox, dtype='float32'), np.array(mbboxes, dtype='float32'),\
               np.array(label_lbbox, dtype='float32'), np.array(lbboxes, dtype='float32')

        # real_boxes = np.array(real_boxes)
        # real_boxes = real_boxes[:, [1, 2, 3, 4, 0]]
        # anchors = np.array(
        #     [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        # num_layers = 3
        # anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        #
        # real_boxes = np.array(real_boxes, dtype='float32')
        # input_shape = np.array((height, width), dtype='int32')
        #
        # boxes_wh = real_boxes[..., 2:4] * input_shape
        #
        # cells = [13, 26, 52]
        # y_true = [np.zeros((cells[n], cells[n], len(anchor_mask[n]), 5 + num_classes), dtype='float32') for n in
        #           range(num_layers)]
        # box_area = boxes_wh[:, 0] * boxes_wh[:, 1]
        #
        # anchor_area = anchors[:, 0] * anchors[:, 1]
        # for r in range(len(real_boxes)):
        #     correct_anchors = []
        #     for anchor in anchors:
        #         correct_anchors.append([min(anchor[0], boxes_wh[r][0]), min(anchor[1], boxes_wh[r][1])])
        #     correct_anchors = np.array(correct_anchors)
        #     correct_anchors_area = correct_anchors[:, 0] * correct_anchors[:, 1]
        #     iou = correct_anchors_area / (box_area[r] + anchor_area - correct_anchors_area)
        #     best_anchor = np.argmax(iou, axis=-1)
        #
        #     for m in range(num_layers):
        #         if best_anchor in anchor_mask[m]:
        #             h = np.floor(real_boxes[r, 0] * cells[m]).astype('int32')
        #             j = np.floor(real_boxes[r, 1] * cells[m]).astype('int32')
        #             k = anchor_mask[m].index(int(best_anchor))
        #             c = real_boxes[r, 4].astype('int32')
        #             y_true[m][j, h, k, 0:4] = real_boxes[r, 0:4]
        #             y_true[m][j, h, k, 4] = 0 if zero_boxes_flag else 1
        #             y_true[m][j, h, k, 5 + c] = 0 if zero_boxes_flag else 1
        #
        # return np.array(y_true[0]), np.array(y_true[1]), np.array(y_true[2])

    def create_scaler(self):

        pass

    def create_tokenizer(self, mode: str, iteration: int, **options):

        """

        Args:
            mode: str
                Режим input/output.
            iteration: int
                Номер входа или выхода.
            **options: Параметры токенайзера:
                       num_words: int
                           Количество слов для токенайзера.
                       filters: str
                           Символы, подлежащие удалению.
                       lower: bool
                           Перевод заглавных букв в строчные.
                       split: str
                           Символ разделения.
                       char_level: bool
                           Учёт каждого символа в качестве отдельного токена.
                       oov_token: str
                           В случае указания этот токен будет заменять все слова, не попавшие в
                           диапазон частотности слов 0 < num_words.

        Returns:
            Объект Токенайзер.

        """

        self.tokenizer[f'{mode}_{iteration}'] = Tokenizer(**options)

        pass

    def create_word2vec(self, mode: str, iteration: int, words: list, **options) -> None:

        """

        Args:
            mode: str
                Режим input/output.
            iteration: int
                Номер входа или выхода.
            words: list
                Список слов для обучения Word2Vec.
            **options: Параметры Word2Vec:
                       size: int
                           Dimensionality of the word vectors.
                       window: int
                           Maximum distance between the current and predicted word within a sentence.
                       min_count: int
                           Ignores all words with total frequency lower than this.
                       workers: int
                           Use these many worker threads to train the model (=faster training with multicore machines).
                       iter: int
                           Number of iterations (epochs) over the corpus.

        Returns:
            Объект Word2Vec.

        """

        self.word2vec[f'{mode}_{iteration}'] = Word2Vec(words, **options)

        pass

    def inverse_data(self, put: str, array: np.ndarray):

        """

        Args:
            put: str
                Рассматриваемый вход или выход (input_2, output_1);
            array: np.ndarray
                NumPy массив, подлежащий возврату в исходное состояние.

        Returns:
            Данные в исходном состоянии.

        """

        inverted_data = None

        for attr in self.__dict__.keys():
            if self.__dict__[attr] and put in self.__dict__[attr].keys():
                if attr == 'tokenizer':
                    if array.shape[0] == self.tokenizer[put].num_words:
                        idx = 0
                        arr = []
                        for num in array:
                            if num == 1:
                                arr.append(idx)
                            idx += 1
                        array = np.array(arr)
                    inv_tokenizer = {index: word for word, index in self.tokenizer[put].word_index.items()}
                    inverted_data = ' '.join([inv_tokenizer[seq] for seq in array])

                elif attr == 'word2vec':
                    text_list = []
                    for i in range(len(array)):
                        text_list.append(
                            self.word2vec[put].wv.most_similar(positive=np.expand_dims(array[i], axis=0), topn=1)[0][0])
                    inverted_data = ' '.join(text_list)

                elif attr == 'scaler':
                    original_shape = array.shape
                    array = array.reshape(-1, 1)
                    array = self.scaler[put].inverse_transform(array)
                    inverted_data = array.reshape(original_shape)
            break

        return inverted_data


class Preprocessing(object):

    def __init__(self):

        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}

    def create_scaler(self):

        pass

    def create_tokenizer(self, mode: str, iteration: int, **options):

        """

        Args:
            mode: str
                Режим input/output.
            iteration: int
                Номер входа или выхода.
            **options: Параметры токенайзера:
                       num_words: int
                           Количество слов для токенайзера.
                       filters: str
                           Символы, подлежащие удалению.
                       lower: bool
                           Перевод заглавных букв в строчные.
                       split: str
                           Символ разделения.
                       char_level: bool
                           Учёт каждого символа в качестве отдельного токена.
                       oov_token: str
                           В случае указания этот токен будет заменять все слова, не попавшие в
                           диапазон частотности слов 0 < num_words.

        Returns:
            Объект Токенайзер.

        """

        self.tokenizer[f'{mode}_{iteration}'] = Tokenizer(**options)

        pass

    def create_word2vec(self, mode: str, iteration: int, x_word: list, **options) -> None:

        """

        Args:
            mode: str
                Режим input/output.
            iteration: int
                Номер входа или выхода.
            x_word: list
                Список слов для обучения Word2Vec.
            **options: Параметры Word2Vec:
                       size: int
                           Dimensionality of the word vectors.
                       window: int
                           Maximum distance between the current and predicted word within a sentence.
                       min_count: int
                           Ignores all words with total frequency lower than this.
                       workers: int
                           Use these many worker threads to train the model (=faster training with multicore machines).
                       iter: int
                           Number of iterations (epochs) over the corpus.

        Returns:
            Объект Word2Vec.

        """

        self.word2vec[f'{mode}_{iteration}'] = Word2Vec(x_word, **options)

        pass

    def inverse_data(self, put: str, array: np.ndarray):

        """

        Args:
            put: str
                Рассматриваемый вход или выход (input_2, output_1);
            array: np.ndarray
                NumPy массив, подлежащий возврату в исходное состояние.

        Returns:
            Данные в исходном состоянии.

        """

        inverted_data = None

        for attr in self.__dict__.keys():
            if self.__dict__[attr] and put in self.__dict__[attr].keys():
                if attr == 'tokenizer':
                    if array.shape[0] == self.tokenizer[put].num_words:
                        idx = 0
                        arr = []
                        for num in array:
                            if num == 1:
                                arr.append(idx)
                            idx += 1
                        array = np.array(arr)
                    inv_tokenizer = {index: word for word, index in self.tokenizer[put].word_index.items()}
                    inverted_data = ' '.join([inv_tokenizer[seq] for seq in array])

                elif attr == 'word2vec':
                    text_list = []
                    for i in range(len(array)):
                        text_list.append(
                            self.word2vec[put].wv.most_similar(positive=np.expand_dims(array[i], axis=0), topn=1)[0][0])
                    inverted_data = ' '.join(text_list)

                elif attr == 'scaler':
                    original_shape = array.shape
                    array = array.reshape(-1, 1)
                    array = self.scaler[put].inverse_transform(array)
                    inverted_data = array.reshape(original_shape)
            break

        return inverted_data


class PrepareDTS(object):

    def __init__(self, trds_path='/content/drive/MyDrive/TerraAI/datasets'):

        self.name: str = ''
        self.source: str = ''
        self.language = None
        self.trds_path: str = trds_path
        self.input_shape: dict = {}
        self.input_dtype: dict = {}
        self.input_datatype: str = ''
        self.input_names: dict = {}
        self.output_shape: dict = {}
        self.output_dtype: dict = {}
        self.output_datatype: dict = {}
        self.output_names: dict = {}
        self.split_sequence: dict = {}
        self.file_folder: str = ''
        self.use_generator: bool = False
        self.zip_params: dict = {}
        self.instructions: dict = {'inputs': {}, 'outputs': {}}
        self.tags: dict = {}
        self.task_type: dict = {}
        self.one_hot_encoding: dict = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.dts_prepared: bool = False

        self.dataloader = None
        self.createarray = CreateArray()

        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.dataset: dict = {}

        pass

    @staticmethod
    def _set_datatype(shape) -> str:

        datatype = {0: 'DIM',
                    1: 'DIM',
                    2: 'DIM',
                    3: '1D',
                    4: '2D',
                    5: '3D'
                    }

        return datatype[len(shape)]

    @staticmethod
    def _set_language(name: str):

        language = {'imdb': 'English',
                    'boston_housing': 'English',
                    'reuters': 'English',
                    'заболевания': 'Russian',
                    'договоры': 'Russian',
                    'умный_дом': 'Russian',
                    'квартиры': 'Russian'
                    }

        if name in language.keys():
            return language[name]
        else:
            return None

    def generator_train(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['train']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(6):
                        outputs[f'output_{int(key[-1]) + i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

            yield inputs, outputs

    def generator_val(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['val']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(6):
                        outputs[f'output_{int(key[-1]) + i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

            yield inputs, outputs

    def generator_test(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['test']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(6):
                        outputs[f'output_{int(key[-1]) + i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

            yield inputs, outputs

    def keras_datasets(self, dataset: str, **options):

        self.name = dataset.lower()
        tags = {'mnist': {'input_1': 'images', 'output_1': 'classification'},
                'fashion_mnist': {'input_1': 'images', 'output_1': 'classification'},
                'cifar10': {'input_1': 'images', 'output_1': 'classification'},
                'cifar100': {'input_1': 'images', 'output_1': 'classification'},
                'imdb': {'input_1': 'text', 'output_1': 'classification'},
                'boston_housing': {'input_1': 'text', 'output_1': 'regression'},
                'reuters': {'input_1': 'text', 'output_1': 'classification'}}
        self.tags = tags[self.name]
        self.source = 'tensorflow.keras'
        data = {
            'mnist': mnist,
            'fashion_mnist': fashion_mnist,
            'cifar10': cifar10,
            'cifar100': cifar100,
            'imdb': imdb,
            'reuters': reuters,
            'boston_housing': boston_housing
        }
        (x_train, y_train), (x_val, y_val) = data[self.name].load_data()

        self.language = self._set_language(self.name)
        if 'classification' in self.tags['output_1']:
            self.num_classes['output_1'] = len(np.unique(y_train, axis=0))
            if self.name == 'fashion_mnist':
                self.classes_names['output_1'] = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                                  'Shirt',
                                                  'Sneaker', 'Bag', 'Ankle boot']
            elif self.name == 'cifar10':
                self.classes_names['output_1'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                                                  'horse', 'ship',
                                                  'truck']
            else:
                self.classes_names['output_1'] = [str(i) for i in range(len(np.unique(y_train, axis=0)))]
        else:
            self.num_classes['output_1'] = 1

        if 'net' in options.keys() and self.name in list(data.keys())[:4]:
            if options['net'].lower() == 'linear':
                x_train = x_train.reshape((-1, np.prod(np.array(x_train.shape)[1:])))
                x_val = x_val.reshape((-1, np.prod(np.array(x_val.shape)[1:])))
            elif options['net'].lower() == 'conv':
                if len(x_train.shape) == 3:
                    x_train = x_train[..., None]
                    x_val = x_val[..., None]

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

            shape_xt = x_train.shape
            shape_xv = x_val.shape
            x_train = x_train.reshape(-1, 1)
            x_val = x_val.reshape(-1, 1)

            if options['scaler'] == 'MinMaxScaler':
                self.createarray.scaler['input_1'] = MinMaxScaler()
                if 'classification' not in self.tags['output_1']:
                    self.createarray.scaler['output_1'] = MinMaxScaler()

            elif options['scaler'] == 'StandardScaler':
                self.createarray.scaler['input_1'] = StandardScaler()
                if 'classification' not in self.tags['output_1']:
                    self.createarray.scaler['output_1'] = StandardScaler()

            self.createarray.scaler['input_1'].fit(x_train)
            x_train = self.createarray.scaler['input_1'].transform(x_train)
            x_val = self.createarray.scaler['input_1'].transform(x_val)
            x_train = x_train.reshape(shape_xt)
            x_val = x_val.reshape(shape_xv)

            if 'classification' not in self.tags['output_1']:
                shape_yt = y_train.shape
                shape_yv = y_val.shape
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
                self.createarray.scaler['output_1'].fit(y_train)
                y_train = self.createarray.scaler['output_1'].transform(y_train)
                y_val = self.createarray.scaler['output_1'].transform(y_val)
                y_train = y_train.reshape(shape_yt)
                y_val = y_val.reshape(shape_yv)
        else:
            self.createarray.scaler['output_1'] = None

        self.one_hot_encoding['output_1'] = False
        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] is True:
            if 'classification' in self.tags['output_1']:
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)), dtype='uint8')
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)), dtype='uint8')
                self.one_hot_encoding['output_1'] = True

        self.input_shape['input_1'] = x_train.shape if len(x_train.shape) < 2 else x_train.shape[1:]
        self.input_datatype = self._set_datatype(shape=x_train.shape)
        self.input_names['input_1'] = 'Вход'
        self.output_shape['output_1'] = y_train.shape[1:]
        self.output_datatype['output_1'] = self._set_datatype(shape=y_train.shape)
        self.output_names['input_1'] = 'Выход'

        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=True)
        self.X['train']['input_1'] = x_train
        self.X['val']['input_1'] = x_val
        self.X['test']['input_1'] = x_test
        self.Y['train']['output_1'] = y_train
        self.Y['val']['output_1'] = y_val
        self.Y['test']['output_1'] = y_test

        self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
        self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
        self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        return self

    def prepare_dataset(self, dataset_name: str, source: str):

        def load_arrays():

            for sample in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays')):
                for arr in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample)):
                    if 'input' in arr:
                        self.X[sample][arr[:arr.rfind('.')]] = joblib.load(
                            os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample, arr))
                    elif 'output' in arr:
                        self.Y[sample][arr[:arr.rfind('.')]] = joblib.load(
                            os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample, arr))

            pass

        def load_scalers():

            scalers = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'scalers')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    scalers.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in scalers:
                    self.createarray.scaler[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.scaler[put] = None

            pass

        def load_tokenizer():

            tokenizer = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'tokenizer')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    tokenizer.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in tokenizer:
                    self.createarray.tokenizer[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.tokenizer[put] = None

            pass

        def load_word2vec():

            word2v = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'word2vec')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    word2v.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in word2v:
                    self.createarray.word2vec[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.word2vec[put] = None

            pass

        def load_augmentation():

            augmentation = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'augmentation')
            if os.path.exists(folder_path):
                for aug in os.listdir(folder_path):
                    augmentation.append(aug[:-3])

            for put in list(self.tags.keys()):
                if put in augmentation:
                    self.createarray.augmentation[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.augmentation[put] = None

            pass

        if dataset_name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters'] and \
                source != 'custom_dataset':
            if dataset_name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(dataset_name, one_hot_encoding=True, scaler='MinMaxScaler', net='conv')
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'imdb':
                self.keras_datasets(dataset_name, one_hot_encoding=True)
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'reuters':
                self.keras_datasets(dataset_name)
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'boston_housing':
                self.keras_datasets(dataset_name, scaler='StandardScaler')
                self.task_type['output_1'] = 'regression'
        elif source == 'custom_dataset':
            with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'config.json'), 'r') as cfg:
                data = json.load(cfg)
            for key, value in data.items():
                self.__dict__[key] = value
            if self.use_generator:
                if 'text' in self.tags.values():
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'txt_list.json'),
                              'r') as txt:
                        self.createarray.txt_list = json.load(txt)

                self.dataloader = Dataloader()
                self.dataloader.load_data(strict_object=SourceData(**self.zip_params))

                with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'sequence.json'),
                          'r') as cfg:
                    self.split_sequence = json.load(cfg)
                for inp in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions',
                                                   'inputs')):
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'inputs', inp),
                              'r') as cfg:
                        data = json.load(cfg)
                    self.instructions['inputs'][inp[:inp.rfind('.')]] = data
                for out in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions',
                                                   'outputs')):
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'outputs', out),
                              'r') as cfg:
                        data = json.load(cfg)
                    self.instructions['outputs'][out[:out.rfind('.')]] = data
                self.createarray.file_folder = self.dataloader.file_folder

                self.dataset['train'] = Dataset.from_generator(self.generator_train,
                                                               output_shapes=(self.input_shape, self.output_shape),
                                                               output_types=(self.input_dtype, self.output_dtype))
                self.dataset['val'] = Dataset.from_generator(self.generator_val,
                                                             output_shapes=(self.input_shape, self.output_shape),
                                                             output_types=(self.input_dtype, self.output_dtype))
                self.dataset['test'] = Dataset.from_generator(self.generator_test,
                                                              output_shapes=(self.input_shape, self.output_shape),
                                                              output_types=(self.input_dtype, self.output_dtype))
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        load_scalers()
        load_tokenizer()
        load_word2vec()
        load_augmentation()

        self.dts_prepared = True

        pass
