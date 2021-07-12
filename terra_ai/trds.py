from terra_ai.data.datasets.extra import SourceModeChoice
from terra_ai.data.datasets.creation import SourceData

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import utils
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
from io import open as io_open
from terra_ai.guiexchange import Exchange
import joblib
import requests
from tempfile import mkdtemp
from datetime import datetime
from pytz import timezone
import json
# import cv2

tr2dj_obj = Exchange()

__version__ = 1.005


class CreateDTS(object):

    def __init__(self, trds_path='/content/drive/MyDrive/TerraAI/datasets',
                 exch_obj=tr2dj_obj):

        self.Exch = exch_obj
        self.django_flag = False
        if self.Exch.property_of != 'TERRA':
            self.django_flag = True

        self.dataloader = Dataloader()
        self.createarray = None

        self.django_flag: bool = False
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
            self.instructions['inputs'][inp] = getattr(self, f"instructions_{self.tags[inp]}")(
                **dataset_dict['inputs'][inp]['parameters'])
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
            if 'txt_list' in self.createarray.__dict__.keys():
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
                y: list = []
                for i in range(self.limit):
                    y.append(getattr(self.createarray, f"create_{self.tags[key]}")(
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

        if self.createarray.scaler:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'scalers'), exist_ok=True)
        if self.createarray.tokenizer:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer'), exist_ok=True)
        if self.createarray.word2vec:
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec'), exist_ok=True)
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
        if options['folder_name']:
            for file_name in sorted(os.listdir(os.path.join(self.file_folder, options['folder_name']))):
                instr.append(os.path.join(options['folder_name'], file_name))
                peg_idx += 1
                y_cls.append(cls_idx)
            self.peg.append(peg_idx)
        else:
            tree = os.walk(self.file_folder)
            for directory, folder, file_name in sorted(tree):
                if bool(file_name) is not False:
                    folder_name = directory.split(os.path.sep)[-1]
                    for name in sorted(file_name):
                        instr.append(os.path.join(folder_name, name))
                        peg_idx += 1
                        y_cls.append(cls_idx)
                    cls_idx += 1
                    self.peg.append(peg_idx)
                else:
                    continue
        instructions['instructions'] = instr
        instructions['parameters'] = options
        self.y_cls = y_cls

        return instructions

    def instructions_video(self):

        pass

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

        self.createarray.create_tokenizer(self.mode, self.iter, **{'num_words': options['max_words_count'],
                                                                   'filters': filters,
                                                                   'lower': True,
                                                                   'split': ' ',
                                                                   'char_level': False,
                                                                   'oov_token': '<UNK>'})
        self.createarray.tokenizer[f'{self.mode}_{self.iter}'].fit_on_texts(list(txt_list.values()))

        self.createarray.txt_list[f'{self.mode}_{self.iter}'] = {}
        for key, value in txt_list.items():
            self.createarray.txt_list[f'{self.mode}_{self.iter}'][key] =\
                self.createarray.tokenizer[f'{self.mode}_{self.iter}'].texts_to_sequences([value])[0]

        if options['word_to_vec']:
            reverse_tok = {}
            for key, value in self.createarray.tokenizer[f'{self.mode}_{self.iter}'].word_index.items():
                reverse_tok[value] = key
            words = []
            for key in self.createarray.txt_list[f'{self.mode}_{self.iter}'].keys():
                for lst in self.createarray.txt_list[f'{self.mode}_{self.iter}'][key]:
                    tmp = []
                    for word in lst:
                        tmp.append(reverse_tok[word])
                    words.append(tmp)
            self.createarray.create_word2vec(mode=self.mode, iteration=self.iter, words=words,
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
            for key in sorted(self.createarray.txt_list[f'{self.mode}_{self.iter}'].keys()):
                index = 0
                while index + length <= len(self.createarray.txt_list[f'{self.mode}_{self.iter}'][key]):
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

        if strict_object.mode == SourceModeChoice.terra:
            self.load_from_terra(strict_object.value)
        elif strict_object.mode == SourceModeChoice.url:
            self.load_from_url(strict_object.value)
        elif strict_object.mode == SourceModeChoice.google_drive:
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

        self.file_folder = None
        self.txt_list: dict = {}

        for key, value in options.items():
            self.__dict__[key] = value

    def create_images(self, image_path: str, **options):

        shape = (options['height'], options['width'])
        img = load_img(path=os.path.join(self.file_folder, image_path), target_size=shape)
        array = img_to_array(img, dtype=np.uint8)
        if options['net'] == 'Linear':
            array = array.reshape(np.prod(np.array(array.shape)))

        return array

    def create_video(self):

        pass

    def create_text(self, sample: dict, **options):

        """

        Args:
            sample: dict
                - file: Название файла.
                - slice: Индексы рассматриваемой части последовательности
            **options: Параметры обработки текста:
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

        filepath: str = sample['file']
        slicing: list = sample['slice']
        array = self.txt_list[options['put']][filepath][slicing[0]:slicing[1]]

        for key, value in options.items():
            if value:
                if key == 'bag_of_words':
                    array = self.tokenizer[options['put']].sequences_to_matrix([array]).astype('uint16')
                elif key == 'word_to_vec':
                    reverse_tok = {}
                    words_list = []
                    for word, index in self.tokenizer[options['put']].word_index.items():
                        reverse_tok[index] = word
                    for idx in array:
                        words_list.append(reverse_tok[idx])
                    array = []
                    for word in words_list:
                        array.append(self.word2vec[options['put']].wv[word])
                break

        array = np.array(array)

        return array

    def create_audio(self):

        pass

    def create_dataframe(self):

        pass

    def create_classification(self, index, **options):

        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        index = np.array(index)

        return index

    def create_regression(self):

        pass

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
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))
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
                for idx, arr in enumerate(
                        os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample))):
                    if 'input' in arr:
                        self.X[sample][f'input_{idx}'] = joblib.load(
                            os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample, arr))
                    elif 'output' in arr:
                        self.Y[sample][f'output_{idx}'] = joblib.load(
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
                load_scalers()
                load_tokenizer()
                load_word2vec()

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
                load_scalers()
                load_tokenizer()
                load_word2vec()
                # load_tsgenerator()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        self.dts_prepared = True

        pass
