from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import utils
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from time import time
from PIL import Image
from librosa import load as librosa_load
import librosa.feature as librosa_feature
import os
import random
import matplotlib.pyplot as plt
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


class DTS(object):

    def __init__(self, f_folder='', path=mkdtemp(), trds_path='/content/drive/MyDrive/TerraAI/datasets',
                 exch_obj=tr2dj_obj):

        self.version = 0.342
        self.Exch = exch_obj
        self.django_flag = False
        if self.Exch.property_of != 'TERRA':
            self.django_flag = True

        self.divide_ratio = [0.8, 0.1, 0.1]
        self.file_folder: str = f_folder
        self.save_path: str = path
        self.trds_path: str = trds_path
        self.name: str = ''
        self.source: str = ''
        self.tags: dict = {}
        self.source_datatype: str = ''
        self.source_shape: dict = {}
        self.input_datatype: str = ''
        self.input_shape: dict = {}
        self.output_datatype: dict = {}
        self.output_shape: dict = {}
        self.one_hot_encoding = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.language: str = ''
        self.dts_prepared: bool = False
        self.task_type: dict = {}
        self.user_parameters: dict = {}
        self.user_tags: list = []

        self.X: dict = {}
        self.Y: dict = {}
        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.df: dict = {}
        self.tsgenerator: dict = {}
        self.tf_dataset: dict = {}

        self.y_cls: np.ndarray = np.array([])
        self.peg: list = []
        self.iter: int = 0
        self.sequences: list = []

        pass

    @staticmethod
    def get_method_parameters(name: str) -> dict:

        method_parameters = {'images': {'folder_name': [''],
                                        'height': 176,
                                        'width': 220,
                                        'net': ['Convolutional', 'Linear'],
                                        'scaler': ['No Scaler', 'MinMaxScaler']},

                             'text': {'folder_name': [''],
                                      'delete_symbols': '',
                                      'x_len': 100,
                                      'step': 30,
                                      'max_words_count': 20000,
                                      'pymorphy': False,
                                      'embedding': True,
                                      'bag_of_words': False,
                                      'word_to_vec': False,
                                      'word_to_vec_size': 200},

                             'audio': {'folder_name': [''],
                                       'length': 22050,
                                       'step': 2205,
                                       'scaler': ['No Scaler', 'StandardScaler', 'MinMaxScaler'],
                                       'audio_signal': True,
                                       'chroma_stft': False,
                                       'mfcc': False,
                                       'rms': False,
                                       'spectral_centroid': False,
                                       'spectral_bandwidth': False,
                                       'spectral_rolloff': False,
                                       'zero_crossing_rate': False},

                             'dataframe': {'file_name': [''],
                                           'separator': '',
                                           'encoding': 'utf-8',
                                           'x_cols': '',
                                           'scaler': ['No Scaler', 'StandardScaler', 'MinMaxScaler']},

                             'classification': {'one_hot_encoding': True},

                             'segmentation': {'folder_name': [''],
                                              'mask_range': 50,
                                              'input_type': ['', 'Ручной ввод', 'Автоматический поиск',
                                                             'Файл аннотации'],
                                              'classes_names': '',
                                              'classes_colors': ''},

                             'text_segmentation': {'open_tags': '',
                                                   'close_tags': ''},

                             'regression': {'y_col': ''},

                             'timeseries': {'length': 1,
                                            'y_cols': '',
                                            'scaler': ['No Scaler', 'StandardScaler', 'MinMaxScaler'],
                                            'task_type': ['timeseries', 'regression']},
                             'object_detection': {}
                             }

        return method_parameters[name]

    def get_parameters_dict(self) -> dict:

        parameters = {}
        list_of_params = ['images', 'text', 'audio', 'dataframe'] + ['classification', 'segmentation',
                                                                     'text_segmentation', 'regression', 'timeseries']

        for elem in list_of_params:
            temp = {}
            for key, value in self.get_method_parameters(elem).items():
                if type(value) == list:
                    if key == 'folder_name' and self.file_folder or key == 'file_name' and self.file_folder:
                        list_folders = ['']
                        list_folders += sorted(os.listdir(self.file_folder))
                        temp[key] = {'type': type(list_folders[0]).__name__,
                                     'default': list_folders[0],
                                     'list': True,
                                     'available': list_folders}
                    else:
                        temp[key] = {'type': type(value[0]).__name__,
                                     'default': value[0],
                                     'list': True,
                                     'available': value}
                elif key == 'classes_names' or key == 'classes_colors':
                    pass
                else:
                    temp[key] = {'type': type(value).__name__,
                                 'default': value}
            parameters[elem] = temp

        return parameters

    def get_datasets_dict(self) -> dict:
        # ['болезни', 'жанры_музыки', 'трафик', 'диалоги']

        datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters', 'sber',
                    'автомобили', 'автомобили_3', 'самолеты', 'губы', 'заболевания', 'договоры', 'умный_дом',
                    'трейдинг', 'квартиры']

        datasets_dict = {}
        for data in datasets:
            datasets_dict[data] = {"tags": [self._set_tag(data), self._set_language(data), self._set_source(data)]}

        return datasets_dict

    def _get_zipfiles(self) -> list:

        return os.listdir(os.path.join(self.trds_path, 'sources'))

    @staticmethod
    def _set_tag(name: str) -> list:

        tags = {'mnist': ['images', 'classification'],
                'fashion_mnist': ['images', 'classification'],
                'cifar10': ['images', 'classification'],
                'cifar100': ['images', 'classification'],
                'imdb': ['text', 'classification'],
                'boston_housing': ['text', 'regression'],
                'reuters': ['text', 'classification'],
                'sber': ['timeseries', 'regression'],
                'автомобили': ['images', 'classification'],
                'автомобили_3': ['images', 'classification'],
                'самолеты': ['images', 'segmentation', 'objectDetection'],
                'умный_дом': ['audio', 'classification', 'smartHome'],
                'договоры': ['text', 'segmentation'],
                'трейдинг': ['traiding', 'timeseries'],
                'квартиры': ['text', 'regression'],
                'болезни': ['images', 'segmentation'],
                'заболевания': ['text', 'classification'],
                'губы': ['images', 'segmentation'],
                'жанры_музыки': ['audio', 'classification']
                }
        return tags[name]

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

    @staticmethod
    def _set_source(name: str) -> str:

        source = {'mnist': 'tensorflow.keras',
                  'fashion_mnist': 'tensorflow.keras',
                  'cifar10': 'tensorflow.keras',
                  'cifar100': 'tensorflow.keras',
                  'imdb': 'tensorflow.keras',
                  'boston_housing': 'tensorflow.keras',
                  'reuters': 'tensorflow.keras',
                  'sber': 'Terra AI',
                  'автомобили': 'Terra AI',
                  'автомобили_3': 'Terra AI',
                  'самолеты': 'Terra AI',
                  'умный_дом': 'Terra AI',
                  'договоры': 'Terra AI',
                  'трейдинг': 'Terra AI',
                  'квартиры': 'Terra AI',
                  'болезни': 'Terra AI',
                  'заболевания': 'Terra AI',
                  'губы': 'Terra AI',
                  'жанры_музыки': 'Terra AI'
                  }

        if name in source.keys():
            return source[name]
        else:
            return 'custom dataset'

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
    def _get_size(start_path) -> str:

        size_bytes = 0
        for path, dirs, files in os.walk(start_path):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        size = ''
        con = 1024
        if 0 <= size_bytes <= con:
            size = f'{size_bytes} B'
        elif con <= size_bytes <= con ** 2:
            size = f'{round(size_bytes / con, 2)} KB'
        elif con ** 2 <= size_bytes <= con ** 3:
            size = f'{round(size_bytes / con ** 2, 2)} MB'
        elif con ** 3 <= size_bytes <= con ** 4:
            size = f'{round(size_bytes / con ** 3, 2)} GB'

        return size

    def _find_colors(self, name: str, num_classes=None, mask_range=None, txt_file=False) -> list:

        if txt_file:
            txt = pd.read_csv(os.path.join(self.file_folder, name), sep=':')
            color_list = {}
            for i in range(len(txt)):
                color_list[txt.loc[i, '# label']] = [int(num) for num in txt.loc[i, 'color_rgb'].split(',')]
        else:
            color_list = []

            for img in sorted(os.listdir(os.path.join(self.file_folder, name))):
                path = os.path.join(self.file_folder, name, img)
                width, height = Image.open(path).size
                img = load_img(path, target_size=(height, width))
                array = img_to_array(img).astype('uint8')

                image = array.reshape(-1, 3)
                km = KMeans(n_clusters=num_classes)
                km.fit(image)
                labels = km.labels_
                cl_cent = np.round(km.cluster_centers_).astype('uint8')[:max(labels) + 1].tolist()
                add_condition = False
                for color in cl_cent:
                    if color_list:
                        if color not in color_list:
                            for in_color in color_list:
                                if color[0] in range(in_color[0] - mask_range, in_color[0] + mask_range) and \
                                        color[1] in range(in_color[1] - mask_range, in_color[1] + mask_range) and \
                                        color[2] in range(in_color[2] - mask_range, in_color[2] + mask_range):
                                    add_condition = False
                                    break
                                else:
                                    add_condition = True
                            if add_condition:
                                color_list.append(color)
                    else:
                        color_list.append(color)
                if len(color_list) >= num_classes:
                    break

        return color_list

    def load_data(self, name: str, mode: str, link=None):

        """
        Create folder and download base in it. Does not change the files original format.
        If base is on the standard base list also print detailed information about specified base.

        Args:
            name (str): name of the base for downloading;

            mode (str): mode of downloading base

            link (str): url where base is located
        """

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

        reference = {
            'трафик': 'Файл в формате csv, содержит информацию о трафике на сайт компании в виде таблицы из двух '
                      'столбцов.\nПервый столбец - дата, второй столбец - объем трафика за сутки.\nВ таблице отражены '
                      'данные за 1095 дней\nСсылка на базу: https://storage.googleapis.com/terra_ai/DataSets/traff.csv',
            'трейдинг': 'Содержит 7 файлов в формате txt с информацией по котировкам акций 3 компаний.\nДанные '
                        'представлены в виде таблицы, отражающей показатели стоимости и объема торгов в '
                        'динамике.\nВсего 9 показателей (компания, дата, время, цена открытия и т.д.)\nСсылка на '
                        'базу: https://storage.googleapis.com/terra_ai/DataSets/shares.zip',
            'автомобили': 'Содержит две папки с изображениями автомобилей в формате png.\nВ первой папке 1088 '
                          'фотографий автомобилей марки Феррари, во второй папке 1161 фотография автомобилей марки '
                          'Мерседес\nСсылка на базу: https://storage.googleapis.com/terra_ai/DataSets/car_2.zip',
            'умный_дом': 'База состоит из звуковых файлов в формате wav для обучения системы "умный дом".\nБаза '
                         'разделена на тренировочную и тестовую выбороки.\nТренировочная выборка включает 4 папки, '
                         '3 из которых содержат  по 104 файла записью одного из типов команд:\nкондиционер, свет, '
                         'телевизор. И 1 папку с записью обычной речи - 50 файлов.\nТестовая выборка содержит запись '
                         'фрагментов речи - 50 файлов \nСсылка на базу: '
                         'https://storage.googleapis.com/terra_ai/DataSets/cHome.zip',
            'квартиры': 'Файл в формате csv, содержит информацию по квартирам, выставленным на продажу.\nДанные '
                        'представлены в виде таблицы из 14 столбцов, отражающих параметры квартиры (станция метро, '
                        'площадь, тип дома, цена и т.д.)\nВ таблице отражены данные по 252 536 квартирам.',
            'диалоги': 'Файл в формате txt, содержит текстовые данные в виде набора блоков "вопрос-ответ".\nКаждый '
                       'блок занимает 2 строки. Всего в файле около 76 тысяч блоков.\nСсылка на базу: '
                       'https://storage.googleapis.com/terra_ai/DataSets/dialog.txt',
            'автомобили_3': 'Содержит три папки с изображениями автомобилей в формате png.\nВ первой папке 1088 '
                            'фотографий Феррари, во второй папке 1161 фотография автомобилей марки Мерседес\nВ третье '
                            'папке 1178 фотографий автомобилей марки Рено.\nСсылка на базу: '
                            'https://storage.googleapis.com/terra_ai/DataSets/car.zip',
            'заболевания': 'База содержит 10 папок с файлами в формате txt с описаниями симптомов 10 заболеваний '
                           'желудочно-кишечного тракта\nСсылка на '
                           'базу:https://storage.googleapis.com/terra_ai/DataSets/symptoms.zip',
            'договоры': 'Содержит 428 файлов в формате txt с текстами договоров.\nСсылка на базу: '
                        'https://storage.googleapis.com/terra_ai/DataSets/docs.zip',
            'самолеты': 'Содержит две папки с изображениями самолетов в формате jpg.\nВ папке "Самолеты" размещен 981 '
                        'файл с исходными изображениями.\nВ папке "Сегменты" размещен 981 файл с сегментированными '
                        'изображениями.\nСсылка на базу: исходные изображения '
                        'https://storage.googleapis.com/terra_ai/DataSets/airplane.zip'
                        ',\n                сегментированные изобажения '
                        'https://storage.googleapis.com/terra_ai/DataSets/segment.zip',
            'болезни': 'Содержит две папки с файлами в формате jpg. \nВ папке ''origin''размещены исходные  '
                       'фотографии, с примерами поражения кожи при 10 дерматологических заболеваниях:\nПсориаз 500 '
                       '\nДерматит 500 \nГерпес 499 \nАкне 510 \nНевус 495 \nВитилиго 504 \nХлоазма 499 \nЭкзема 498 '
                       '\nСыпь 522 \nЛишай 502\nВ папке ''segmentation'' размещены сегментированные '
                       'изображения.\nСсылка на базу: исходные изображения '
                       'https://storage.googleapis.com/terra_ai/DataSets/origin.zip \n                сегментированные '
                       'изображения https://storage.googleapis.com/terra_ai/DataSets/segmentation.zip'
        }

        default_path = self.save_path
        if mode == 'google_drive':
            filepath = os.path.join(self.trds_path, 'sources', name)
            name = name[:name.rfind('.')]
            self.file_folder = os.path.join(default_path, name)
            shutil.unpack_archive(filepath, self.file_folder)
        elif mode == 'url':
            filename = link.split('/')[-1]
            file_folder = pathlib.Path(os.path.join(default_path, filename))
            if '.' in filename:
                name = filename[:filename.rfind('.')]
                file_folder = pathlib.Path(os.path.join(default_path, name))
            self.file_folder = file_folder
            os.makedirs(self.file_folder, exist_ok=True)
            os.makedirs(os.path.join(self.file_folder, 'tmp'), exist_ok=True)
            resp = requests.get(link, stream=True)
            total = int(resp.headers.get('content-length', 0))
            idx = 0
            with open(os.path.join(self.file_folder, 'tmp', filename), 'wb') as out_file, tqdm(
                    desc=f"Загрузка архива {filename}", total=total, unit='iB', unit_scale=True,
                    unit_divisor=1024) as progress_bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = out_file.write(data)
                    progress_bar.update(size)
                    idx += size
                    if self.django_flag:
                        if idx % 143360 == 0 or idx == progress_bar.total:
                            progress_bar_status = \
                                (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                 f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                            if idx == progress_bar.total:
                                self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                            else:
                                self.Exch.print_progress_bar(progress_bar_status)
            if 'zip' in filename or 'zip' in link:
                file_path = pathlib.Path(os.path.join(self.file_folder, 'tmp', filename))
                temp_folder = os.path.join(self.file_folder, 'tmp')
                shutil.unpack_archive(file_path, self.file_folder)
                shutil.rmtree(temp_folder, ignore_errors=True)
        elif mode == 'terra':
            self.language = self._set_language(name=name)
            for base in data[name]:
                self.file_folder = pathlib.Path(default_path).joinpath(name)
                os.makedirs(self.file_folder, exist_ok=True)
                os.makedirs(os.path.join(self.file_folder, 'tmp'), exist_ok=True)
                link = 'https://storage.googleapis.com/terra_ai/DataSets/Numpy/' + base
                resp = requests.get(link, stream=True)
                total = int(resp.headers.get('content-length', 0))
                idx = 0
                with open(os.path.join(self.file_folder, 'tmp', base), 'wb') as out_file, \
                        tqdm(desc=f"Загрузка архива {base}", total=total, unit='iB',
                             unit_scale=True, unit_divisor=1024) as progress_bar:
                    for data in resp.iter_content(chunk_size=1024):
                        size = out_file.write(data)
                        progress_bar.update(size)
                        idx += size
                        if self.django_flag:
                            if idx % 143360 == 0 or idx == progress_bar.total:
                                progress_bar_status = \
                                    (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                     f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                                if idx == progress_bar.total:
                                    self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                                else:
                                    self.Exch.print_progress_bar(progress_bar_status)
                if 'zip' in base:
                    file_path = pathlib.Path(os.path.join(default_path, name, 'tmp', base))
                    temp_folder = self.file_folder.joinpath('tmp')
                    if not temp_folder.exists:
                        os.mkdir(temp_folder)
                    shutil.unpack_archive(file_path, self.file_folder)
                    shutil.rmtree(temp_folder, ignore_errors=True)
            if not self.django_flag:
                if name in reference.keys():
                    print(reference[name])
        self.name = name
        self.source = self._set_source(name)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        return self

    def prepare_dataset(self, **options):

        def load_arrays():

            inp_datatype = []
            for arr in os.listdir(os.path.join(self.file_folder, 'arrays')):
                if 'input' in arr:
                    self.X[arr[:-3]] = joblib.load(os.path.join(self.file_folder, 'arrays', arr))
                    self.input_shape[arr[:-3]] = self.X[arr[:-3]]['data'][0].shape[1:]
                    inp_datatype.append(self._set_datatype(shape=self.X[arr[:-3]]['data'][0].shape))
                elif 'output' in arr:
                    self.Y[arr[:-3]] = joblib.load(os.path.join(self.file_folder, 'arrays', arr))
                    self.output_shape[arr[:-3]] = self.Y[arr[:-3]]['data'][0].shape[1:]
                    if 'object_detection' in self.tags.values():
                        self.output_datatype[arr[:-3]] = '2D'
                    else:
                        self.output_datatype[arr[:-3]] = self._set_datatype(shape=self.Y[arr[:-3]]['data'][0].shape)
            self.input_datatype = ' '.join(inp_datatype)

            pass

        def load_scalers():

            scalers = []
            folder_path = os.path.join(self.file_folder, 'scalers')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    scalers.append(arr[:-3])

            for put in inputs_outputs:
                if put in scalers:
                    self.scaler[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.scaler[put] = None

            pass

        def load_tokenizer():

            tokenizer = []
            folder_path = os.path.join(self.file_folder, 'tokenizer')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    tokenizer.append(arr[:-3])

            for put in inputs_outputs:
                if put in tokenizer:
                    self.tokenizer[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.tokenizer[put] = None

            pass

        def load_word2vec():

            word2v = []
            folder_path = os.path.join(self.file_folder, 'word2vec')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    word2v.append(arr[:-3])

            for put in inputs_outputs:
                if put in word2v:
                    self.word2vec[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.word2vec[put] = None

            pass

        def load_tsgenerator():

            tsgenerator = []
            folder_path = os.path.join(self.file_folder, 'tsgenerator')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    tsgenerator.append(arr[:-3])

            for put in inputs_outputs:
                if put in tsgenerator:
                    self.tsgenerator[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.tsgenerator[put] = None

            pass

        def get_train_generator():

            x_train = {}
            for x_key in self.X.keys():
                x_train[x_key] = self.X[x_key]['data'][0]

            y_train = {}
            for y_key in self.Y.keys():
                y_train[y_key] = self.Y[y_key]['data'][0]

            train_generator = Dataset.from_tensor_slices((x_train, y_train))

            return train_generator

        def get_validation_generator():

            x_val = {}
            for x_key in self.X.keys():
                x_val[x_key] = self.X[x_key]['data'][1]

            y_val = {}
            for y_key in self.Y.keys():
                y_val[y_key] = self.Y[y_key]['data'][1]

            val_generator = Dataset.from_tensor_slices((x_val, y_val))

            return val_generator

        def get_test_generator():

            x_test = {}
            for x_key in self.X.keys():
                x_test[x_key] = self.X[x_key]['data'][2]

            y_test = {}
            for y_key in self.Y.keys():
                y_test[y_key] = self.Y[y_key]['data'][2]

            test_generator = Dataset.from_tensor_slices((x_test, y_test))

            return test_generator

        if options['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing',
                                       'reuters'] and options['source'] != 'custom_dataset':

            if options['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, scaler='MinMaxScaler', net='conv',
                                    test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'imdb':
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'reuters':
                self.keras_datasets(options['dataset_name'], test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'boston_housing':
                self.keras_datasets(options['dataset_name'], scaler='StandardScaler', test=True)
                self.task_type['output_1'] = 'regression'

        else:

            if options['dataset_name'] in ['трейдинг', 'умный_дом', 'квартиры', 'автомобили', 'автомобили_3',
                                           'заболевания', 'договоры', 'самолеты', 'губы', 'sber'] and \
                    options['source'] != 'custom_dataset':

                self.load_data(options['dataset_name'], mode='terra')
                self.file_folder = os.path.join(self.save_path, options['dataset_name'])
            else:
                self.file_folder = os.path.join(self.trds_path, f"dataset {options['dataset_name']}")
            with open(os.path.join(self.file_folder, 'config.json'), 'r') as cfg:
                data = json.load(cfg)
            for key, value in data.items():
                self.__dict__[key] = value
            load_arrays()
            inputs_outputs = list(self.X.keys()) + list(self.Y.keys())
            load_scalers()
            load_tokenizer()
            load_word2vec()
            load_tsgenerator()

        self.tf_dataset['train'] = get_train_generator()
        self.tf_dataset['val'] = get_validation_generator()
        self.tf_dataset['test'] = get_test_generator()

        # temp_attributes = ['df', 'peg', 'user_parameters']
        # for item in temp_attributes:
        #     if hasattr(self, item):
        #         delattr(self, item)
        self.dts_prepared = True

        return self

    def inverse_data(self, put: str, array: np.ndarray):

        if self.tags[put] == 'text':
            if len(array.shape) == 1:
                if array.shape[0] == self.tokenizer[put].num_words:
                    idx = 0
                    arr = []
                    for num in array:
                        if num == 1:
                            arr.append(idx)
                        idx += 1
                    array = np.array(arr)
                inv_tokenizer = {index: word for word, index in self.tokenizer[put].word_index.items()}
                text: str = ' '.join([inv_tokenizer[seq] for seq in array])
            else:
                text_list = []
                for i in range(len(array)):
                    text_list.append(
                        self.word2vec[put].wv.most_similar(positive=np.expand_dims(array[i], axis=0), topn=1)[0][0])
                text: str = ' '.join(text_list)

            return text

    def keras_datasets(self, dataset: str, **options):

        def print_data(name, x, y):

            pics = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
            text = ['imdb', 'reuters', 'boston_housing']

            if name in pics:
                fig, axs = plt.subplots(1, 10, figsize=(25, 3))
                for i in range(10):
                    label_indexes = np.where(y == i)[0]
                    index = random.choice(label_indexes)
                    img = x[index]
                    title = y[index]
                    if name in ['mnist', 'fashion_mnist']:
                        axs[i].imshow(Image.fromarray(img), cmap='gray')
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names["output_1"][title]}')
                    else:
                        axs[i].imshow(Image.fromarray(img))
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names["output_1"][title[0]]}')

            if name in text:
                if name in ['imdb', 'reuters']:
                    pd.DataFrame({'x_train': x, 'y_train': y}).head()
                else:
                    df = pd.DataFrame(x)
                    df['y_train'] = y
                    df.head()

            pass

        x_train: np.ndarray = np.array([])
        y_train: np.ndarray = np.array([])
        x_val: np.ndarray = np.array([])
        y_val: np.ndarray = np.array([])

        cur_time = time()
        self.name = dataset.lower()
        tags = self._set_tag(self.name)
        self.tags = {'input_1': tags[0],
                     'output_1': tags[1]}
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
        progress_bar = tqdm(range(1), ncols=800)
        progress_bar.set_description(f'Загрузка датасета {self.name}')
        idx = 0
        for _ in progress_bar:
            (x_train, y_train), (x_val, y_val) = data[self.name].load_data()
            if self.django_flag:
                idx += 1
                progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                       f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)

        self.source_shape['input_1'] = x_train.shape if len(x_train.shape) < 2 else x_train.shape[1:]
        self.language = self._set_language(self.name)
        self.source_datatype += f' {self._set_datatype(shape=x_train.shape)}'
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

        if not self.django_flag:
            print_data(self.name, x_train, y_train)

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
                self.scaler['input_1'] = MinMaxScaler()
                if 'classification' not in self.tags['output_1']:
                    self.scaler['output_1'] = MinMaxScaler()

            elif options['scaler'] == 'StandardScaler':
                self.scaler['input_1'] = StandardScaler()
                if 'classification' not in self.tags['output_1']:
                    self.scaler['output_1'] = StandardScaler()

            self.scaler['input_1'].fit(x_train)
            x_train = self.scaler['input_1'].transform(x_train)
            x_val = self.scaler['input_1'].transform(x_val)
            x_train = x_train.reshape(shape_xt)
            x_val = x_val.reshape(shape_xv)

            if 'classification' not in self.tags['output_1']:
                shape_yt = y_train.shape
                shape_yv = y_val.shape
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
                self.scaler['output_1'].fit(y_train)
                y_train = self.scaler['output_1'].transform(y_train)
                y_val = self.scaler['output_1'].transform(y_val)
                y_train = y_train.reshape(shape_yt)
                y_val = y_val.reshape(shape_yv)
        else:
            self.scaler['output_1'] = None

        self.one_hot_encoding['output_1'] = False
        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] is True:
            if 'classification' in self.tags['output_1']:
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))
                self.one_hot_encoding['output_1'] = True

        self.input_shape['input_1'] = x_train.shape if len(x_train.shape) < 2 else x_train.shape[1:]
        self.input_datatype = self._set_datatype(shape=x_train.shape)
        self.output_shape['output_1'] = y_train.shape[1:]
        self.output_datatype['output_1'] = self._set_datatype(shape=y_train.shape)

        self.X = {'input_1': {'data_name': 'Вход',
                              'data': (x_train, x_val, None)}}
        self.Y = {'output_1': {'data_name': 'Выход',
                               'data': (y_train, y_val, None)}}

        if 'test' in options.keys() and options['test'] is True:
            split_ratio = self.divide_ratio[1:]
            split_size = min(split_ratio) / sum(split_ratio)
            x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=1 - split_size, shuffle=True)
            self.X['input_1']['data'] = (x_train, x_val, x_test)
            self.Y['output_1']['data'] = (y_train, y_val, y_test)

        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            x_arrays = ['x_train', 'x_val', 'x_test']
            for i, item_x in enumerate(self.X['input_1']['data']):
                if item_x is not None:
                    print(f"Размерность {x_arrays[i]}: {item_x.shape}")
            y_arrays = ['y_train', 'y_val', 'y_test']
            for i, item_y in enumerate(self.Y['output_1']['data']):
                if item_y is not None:
                    print(f"Размерность {y_arrays[i]}: {item_y.shape}")

        return self

    def images(self, **options) -> np.ndarray:

        folder_name: str = options['folder_name']
        height: int = options['height']
        width: int = options['width']
        net: str = options['net']
        scaler: str = options['scaler']

        if folder_name == '':
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]
        shape = (height, width)
        x_array = []
        y_cls = []

        for _, dir_names, filename in sorted(os.walk(working_folder)):

            folders = sorted(dir_names)
            folders_num = len(dir_names) if len(dir_names) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except IndexError:
                    pass

                files = sorted(os.listdir(temp_path))
                for j in range(len(self.user_parameters['out'])):
                    if self.user_parameters['out'][f'output_{j + 1}']['tag'] == 'object_detection':

                        data = {}
                        with open(os.path.join(self.file_folder, 'obj.data'), 'r') as dt:
                            d = dt.read()
                        for elem in d.split('\n'):
                            if elem:
                                elem = elem.split(' = ')
                                data[elem[0]] = elem[1]

                        files = []
                        with open(os.path.join(self.file_folder, data["train"].split("/")[-1]), 'r') as dt:
                            image = dt.read()
                        for elem in image.split('\n'):
                            if elem:
                                files.append(os.path.join(*elem.split('/')[2:]))
                        break

                progress_bar = tqdm(files, ncols=800)
                if folders_num == 1:
                    description = f'Сохранение изображений'
                else:
                    description = f'Сохранение изображений из папки {folders[i]}'
                progress_bar.set_description(description)
                if len(self.source_shape) < 3:
                    source_shape = Image.open(os.path.join(temp_path, os.listdir(temp_path)[0])).size
                    self.source_shape[f'input_{self.iter}'] = (source_shape[1], source_shape[0], 3)
                idx = 0
                for file in progress_bar:
                    img = load_img(os.path.join(temp_path, file), target_size=shape)
                    x_array.append(img_to_array(img).astype('uint8'))
                    y_cls.append(i)
                    idx += 1
                    if self.django_flag:
                        progress_bar_status =\
                            (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                             f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)
                self.peg.append(idx + self.peg[-1])

            break

        x_array = np.array(x_array)
        y_cls = np.array(y_cls)
        self.source_datatype += f' {self._set_datatype(shape=x_array.shape)}'

        if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
            shape_x = x_array.shape
            x_array = x_array.reshape(-1, 1)
            if scaler == 'MinMaxScaler':
                self.scaler[f'input_{self.iter}'] = MinMaxScaler()
            elif scaler == 'StandardScaler':
                self.scaler[f'input_{self.iter}'] = StandardScaler()
            self.scaler[f'input_{self.iter}'].fit(x_array)
            x_array = self.scaler[f'input_{self.iter}'].transform(x_array)
            x_array = x_array.reshape(shape_x)

        if net == 'Linear':
            x_array = x_array.reshape(-1, np.prod(np.array(x_array.shape)[1:]))

        if 'classification' in self.tags.values():
            self.y_cls = y_cls.astype('int')
            del y_cls

        return x_array

    # def video(self, folder_name=[''], height=64, width=64, max_frames_per_class=10000,
    #           scaler=['No Scaler', 'StandardScaler', 'MinMaxScaler']) -> np.ndarray:
    #
    #     if folder_name == None:
    #         folder_name = self.file_folder
    #     else:
    #         folder_name = os.path.join(self.file_folder, folder_name)
    #
    #     X = []
    #     y_cls = []
    #
    #     for _, dir_names, filename in sorted(os.walk(folder_name)):
    #
    #         folders = sorted(dir_names)
    #         folders_num = len(dir_names) if len(dir_names) != 0 else 1
    #         for i in range(folders_num):
    #             temp_path = folder_name
    #             try:
    #                 temp_path = os.path.join(folder_name, folders[i])
    #             except IndexError:
    #                 pass
    #
    #             files = sorted(os.listdir(temp_path))
    #             progress_bar = tqdm(files, ncols=800)
    #             if folders_num == 1:
    #                 description = f'Сохранение видеофайлов'
    #             else:
    #                 description = f'Сохранение видеофайлов из папки {folders[i]}'
    #             progress_bar.set_description(description)
    #             idx = 1
    #             for file in progress_bar:
    #                 vc = cv2.VideoCapture(os.path.join(temp_path, file))
    #                 while True:
    #                     success, frame = vc.read()
    #                     if not success:
    #                         break
    #                     resized_frame = cv2.resize(frame, (height, width))
    #                     X.append(resized_frame)
    #                     y_cls.append(i)
    #                 if self.django_flag:
    #                     idx += 1
    #                     progress_bar_status =\
    #                         (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
    #                          f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
    #                         self.Exch.print_progress_bar(progress_bar_status)
    #         break
    #
    #     X = np.array(X)
    #     y_cls = np.array(y_cls)
    #
    #     if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
    #
    #         shape_x = X.shape
    #         X = X.reshape(-1, 1)
    #
    #         if scaler == 'MinMaxScaler':
    #             self.scaler[f'input_{self.iter}'] = MinMaxScaler()
    #
    #         elif scaler == 'StandardScaler':
    #             self.scaler[f'input_{self.iter}'] = StandardScaler()
    #
    #         self.scaler[f'input_{self.iter}'].fit(X)
    #         X = self.scaler[f'input_{self.iter}'].transform(X)
    #         X = X.reshape(shape_x)
    #
    #     if 'classification' in self.tags.values():
    #         self.y_cls = y_cls.astype('int')
    #         del y_cls
    #
    #     return X

    def text(self, **options) -> np.ndarray:

        def read_text(file_path):

            del_symbols = ['\n', '\t', '\ufeff']
            if delete_symbols:
                del_symbols += delete_symbols.split(' ')
            with io_open(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
                for del_symbol in del_symbols:
                    text = text.replace(del_symbol, ' ')
            for k in range(len(self.user_parameters['out'])):
                if self.user_parameters['out'][f'output_{k + 1}']['tag'] == 'text_segmentation':
                    open_symbol =\
                        self.user_parameters['out'][f'output_{k + 1}']['parameters']['open_tags'].split(' ')[0][0]
                    close_symbol = \
                        self.user_parameters['out'][f'output_{k + 1}']['parameters']['open_tags'].split(' ')[0][-1]
                    text = re.sub(open_symbol, f" {open_symbol}", text)
                    text = re.sub(close_symbol, f"{close_symbol} ", text)
                    break

            return text

        def apply_pymorphy(text) -> list:

            morph = pymorphy2.MorphAnalyzer()
            words = text.split(' ')
            words = [morph.parse(w)[0].normal_form for w in words]

            return words

        def remove_segmentation_tags(text_sequences, tags_index):

            indexes = []
            for text_sequence in text_sequences:
                temp = []
                for ex in text_sequence:
                    if ex not in tags_index:
                        temp.append(ex)
                indexes.append(temp)

            return indexes

        def get_set_from_indexes(word_indexes, length, stride) -> list:

            sample = []
            words_len = len(word_indexes)

            index = 0
            peg_idx = 0
            while index + length <= words_len:
                sample.append(word_indexes[index:index + length])
                index += stride
                peg_idx += 1
            self.peg.append(peg_idx + self.peg[-1])

            return sample

        def create_sets_multi_classes(word_indexes, length, stride):

            classes_x_samples = []
            for w_i in word_indexes:
                classes_x_samples.append(get_set_from_indexes(w_i, length, stride))

            x_samples = []
            y_samples = []

            pb_idx = 0
            pb = tqdm(range(len(word_indexes)), ncols=800)
            pb.set_description('Формирование массивов')
            for t in pb:
                x_t = classes_x_samples[t]
                for k in range(len(x_t)):
                    x_samples.append(x_t[k])
                    y_samples.append(t)
                if self.django_flag:
                    pb_idx += 1
                    pb_status = (pb.desc, str(round(pb_idx / pb.total, 2)),
                                 f'{str(round(pb.last_print_t - pb.start_t, 2))} сек.')
                    self.Exch.print_progress_bar(pb_status)
            x_samples = np.array(x_samples)
            y_samples = np.array(y_samples)

            return x_samples, y_samples

        folder_name: str = options['folder_name']
        delete_symbols: str = options['delete_symbols']
        x_len: int = options['x_len']
        step: int = options['step']
        max_words_count: int = options['max_words_count']
        pymorphy: bool = options['pymorphy']
        bag_of_words: bool = options['bag_of_words']
        word_to_vec: bool = options['word_to_vec']
        if word_to_vec:
            word_to_vec_size: int = options['word_to_vec_size']

        tags_list: list = []
        txt_list: list = []
        y_array = []

        if folder_name == '':
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]

        for _, dir_names, filename in sorted(os.walk(working_folder)):

            folders = sorted(dir_names)
            folders_num = len(dir_names) if len(dir_names) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except IndexError:
                    pass

                files = sorted(os.listdir(temp_path))
                progress_bar = tqdm(files, ncols=800)
                if folders_num == 1:
                    description = f'Сохранение текстов'
                else:
                    description = f'Сохранение текстов из папки {folders[i]}'
                progress_bar.set_description(description)
                idx = 0
                several_files = False
                for file in progress_bar:
                    if progress_bar.total > 1:
                        if not several_files:
                            txt_list.append(read_text(os.path.join(temp_path, file)))
                            several_files = True
                        else:
                            txt_list[-1] += read_text(os.path.join(temp_path, file))
                    else:
                        txt_list.append(read_text(os.path.join(temp_path, file)))
                    if self.django_flag:
                        idx += 1
                        progress_bar_status =\
                            (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                             f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)
            break

        if pymorphy:
            for i in range(len(txt_list)):
                txt_list[i] = apply_pymorphy(txt_list[i])

        filters = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'text_segmentation':
                open_tags = self.user_parameters['out'][f'output_{i + 1}']['parameters']['open_tags']
                close_tags = self.user_parameters['out'][f'output_{i + 1}']['parameters']['close_tags']
                tags = f'{open_tags} {close_tags}'
                tags_list = tags.split(' ')
                for ch in filters:
                    if ch in set(tags):
                        filters = filters.replace(ch, '')
                break

        tokenizer = Tokenizer(num_words=max_words_count, filters=filters,
                              lower=True, split=' ', char_level=False, oov_token='<UNK>')
        tokenizer.fit_on_texts(txt_list)
        text_seq = tokenizer.texts_to_sequences(txt_list)
        self.sequences = text_seq
        self.tokenizer[f'input_{self.iter}'] = tokenizer

        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'text_segmentation':
                tags_indexes = np.array([tokenizer.word_index[k] for k in tags_list])
                text_seq = remove_segmentation_tags(text_seq, tags_indexes)

        if word_to_vec:
            reverse_tok = {}
            for key, value in tokenizer.word_index.items():
                reverse_tok[value] = key
            x_word = []
            for j, lst in enumerate(text_seq):
                tmp_x = []
                tmp_y = []
                for word in lst:
                    tmp_x.append(reverse_tok[word])
                    tmp_y.append(j)
                x_word.append(tmp_x)
                y_array.append(tmp_y)
            self.word2vec[f'input_{self.iter}'] = Word2Vec(x_word, size=word_to_vec_size, window=10,
                                                           min_count=1, workers=10, iter=10)
            words_vectors = []
            for i, sequences in enumerate(text_seq):
                tmp = []
                for seq in sequences:
                    tmp.append(self.word2vec[f'input_{self.iter}'][reverse_tok[seq]])
                words_vectors.append(tmp)
            x_array, y_array = create_sets_multi_classes(words_vectors, x_len, step)
        else:
            x_array, y_array = create_sets_multi_classes(text_seq, x_len, step)
            if bag_of_words:
                x_array = np.array(tokenizer.sequences_to_matrix(x_array.tolist()))

        self.source_shape[f'input_{self.iter}'] = x_array.shape[1:]
        self.source_datatype += f' {self._set_datatype(shape=x_array.shape)}'

        if 'classification' in self.tags.values():
            self.y_cls = y_array.astype('int')
            del y_array

        return x_array

    def dataframe(self, **options) -> np.ndarray:

        file_name: str = options['file_name']
        separator: str = options['separator']
        encoding: str = options['encoding']
        x_cols: str = options['x_cols']
        scaler: str = options['scaler']

        self.classes_names[f'input_{self.iter}'] = x_cols.split(' ')
        if separator:
            dataframe = pd.read_csv(os.path.join(self.file_folder, file_name), sep=separator, encoding=encoding)
        else:
            dataframe = pd.read_csv(os.path.join(self.file_folder, file_name), encoding=encoding)
        self.df[f'input_{self.iter}'] = dataframe

        x_array = dataframe[x_cols.split(' ')].to_numpy()

        self.source_shape[f'input_{self.iter}'] = x_array.shape[1:]
        self.source_datatype += f' {self._set_datatype(shape=x_array.shape)}'

        if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
            shape_x = x_array.shape
            x_array = x_array.reshape(-1, 1)
            if scaler == 'MinMaxScaler':
                self.scaler[f'input_{self.iter}'] = MinMaxScaler()
            elif scaler == 'StandardScaler':
                self.scaler[f'input_{self.iter}'] = StandardScaler()
            self.scaler[f'input_{self.iter}'].fit(x_array)
            x_array = self.scaler[f'input_{self.iter}'].transform(x_array)
            x_array = x_array.reshape(shape_x)

        # Если надо работать с временными рядами
        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'timeseries':
                length = self.user_parameters['out'][f'output_{i + 1}']['parameters']['length']
                scaler = self.user_parameters['out'][f'output_{i + 1}']['parameters']['scaler']
                y_cols = self.user_parameters['out'][f'output_{i + 1}']['parameters']['y_cols']
                y_array = self.df[f'input_{self.iter}'][y_cols.split(' ')].to_numpy()
                if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
                    shape_y = y_array.shape
                    y_array = y_array.reshape(-1, 1)
                    if scaler == 'MinMaxScaler':
                        self.scaler[f'output_{i + 1}'] = MinMaxScaler()
                    elif scaler == 'StandardScaler':
                        self.scaler[f'output_{i + 1}'] = StandardScaler()
                    self.scaler[f'output_{i + 1}'].fit(y_array)
                    y_array = self.scaler[f'output_{i + 1}'].transform(y_array)
                    y_array = y_array.reshape(shape_y)

                generator = TimeseriesGenerator(x_array, y_array, length=length, stride=1, batch_size=1)
                self.tsgenerator[f'input_{self.iter}'] = generator
                x_array = []
                for j in range(len(self.tsgenerator[f'input_{self.iter}'])):
                    for k in range(len(self.tsgenerator[f'input_{self.iter}'][j][0])):
                        x_array.append(self.tsgenerator[f'input_{self.iter}'][j][0][k])  # Записываем каждый батч
                x_array = np.array(x_array)
            break

        return x_array

    def regression(self, **options) -> np.ndarray:

        y_col: str = options['y_col']
        y_col: list = y_col.split(' ')

        y_array = np.array([])
        self.classes_names[f'output_{self.iter}'] = y_col
        self.num_classes[f'output_{self.iter}'] = len(y_col)

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'dataframe':

                y_array = self.df[f'input_{i + 1}'][y_col].to_numpy()
                if self.user_parameters['inp'][f'input_{i + 1}']['parameters']['scaler'] in ['MinMaxScaler',
                                                                                             'StandardScaler']:
                    y_shape = y_array.shape
                    y_array = y_array.reshape(-1, 1)
                    y_array = self.scaler[f'input_{i + 1}'].transform(y_array)
                    y_array = y_array.reshape(y_shape)

                break

        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.peg = [0]
        for ratio in self.divide_ratio[:-1]:
            self.peg.append(self.peg[-1] + int(round(len(y_array) * ratio, 0)))
        self.peg.append(len(y_array))
        self.task_type[f'output_{self.iter}'] = 'regression'

        return y_array

    def timeseries(self, **options) -> np.ndarray:

        y_array = []
        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'dataframe':
                for j in range(len(self.tsgenerator[f'input_{i + 1}'])):
                    for k in range(len(self.tsgenerator[f'input_{i + 1}'][j][1])):
                        y_array.append(self.tsgenerator[f'input_{i + 1}'][j][1][k])
                y_array = np.array(y_array)
            break

        y_cols: str = options['y_cols']
        task_type: str = options['task_type']

        self.classes_names[f'output_{self.iter}'] = y_cols.split(' ')
        self.num_classes[f'output_{self.iter}'] = len(y_cols.split(' '))
        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.peg = [0]
        for ratio in self.divide_ratio[:-1]:
            self.peg.append(self.peg[-1] + int(round(len(y_array) * ratio, 0)))
        self.peg.append(len(y_array))
        self.task_type[f'output_{self.iter}'] = task_type

        return y_array

    def audio(self, **options):

        def call_librosa(feature, section, sr):

            array: np.ndarray = np.array([])

            if feature in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                array = getattr(librosa_feature, feature)(y=section, sr=sr)
            elif feature == 'rms':
                array = getattr(librosa_feature, feature)(y=section)[0]
            elif feature == 'zero_crossing_rate':
                array = getattr(librosa_feature, feature)(y=section)

            return array

        def wav_to_features(section, sr):

            for feature in feature_dict.keys():
                if feature == 'audio_signal':
                    feature_dict['audio_signal'].append(section)
                else:
                    feature_dict[feature].append(call_librosa(feature, section, sr))

            pass

        folder_name: str = options['folder_name']
        length: int = options['length']
        step: int = options['step']
        scaler: str = options['scaler']
        audio_signal: bool = options['audio_signal']
        chroma_stft: bool = options['chroma_stft']
        mfcc: bool = options['mfcc']
        rms: bool = options['rms']
        spectral_centroid: bool = options['spectral_centroid']
        spectral_bandwidth: bool = options['spectral_bandwidth']
        spectral_rolloff: bool = options['spectral_rolloff']
        zero_crossing_rate: bool = options['zero_crossing_rate']

        features_str = ['audio_signal', 'chroma_stft', 'mfcc', 'rms', 'spectral_centroid', 'spectral_bandwidth',
                        'spectral_rolloff', 'zero_crossing_rate']
        features = [audio_signal, chroma_stft, mfcc, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff,
                    zero_crossing_rate]
        feature_dict = {}
        for i, ft in enumerate(features):
            if ft is True:
                feature_dict[features_str[i]] = []

        if folder_name == '':
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]
        y_array = []

        for _, dir_names, filename in sorted(os.walk(working_folder)):

            folders = sorted(dir_names)
            folders_num = len(dir_names) if len(dir_names) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except IndexError:
                    pass

                files = [os.path.join(temp_path, wav_file) for wav_file in
                         sorted(os.listdir(temp_path))]

                if folders_num == 1:
                    description = f'Сохранение аудиофайлов'
                else:
                    description = f'Сохранение аудиофайлов из папки {folders[i]}'

                progress_bar = tqdm(files, ncols=800)
                progress_bar.set_description(description)
                idx = 0
                peg_idx = 0
                for file in progress_bar:
                    y, sample_rate = librosa_load(file)
                    while len(y) >= length:
                        sect = y[:length]
                        sect = np.array(sect)
                        wav_to_features(sect, sample_rate)
                        y = y[step:]
                        y_array.append(i)
                        peg_idx += 1
                    idx += 1
                    if self.django_flag:
                        progress_bar_status =\
                            (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                             f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)
                self.peg.append(peg_idx + self.peg[-1])
            break

        y_array = np.array(y_array)
        for ft in feature_dict.keys():
            self.iter += 1
            self.X[f'input_{self.iter}'] = {'data_name': ft, 'data': np.array(feature_dict[ft])}
            self.source_shape[f'input_{self.iter}'] = self.X[f'input_{self.iter}']['data'].shape[1:]
            self.source_datatype += f' {self._set_datatype(shape=self.X[f"input_{self.iter}"]["data"].shape)}'

            if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':

                shape_x = self.X[f'input_{self.iter}']['data'].shape
                x_array = self.X[f'input_{self.iter}']['data'].reshape(-1, 1)
                if scaler == 'MinMaxScaler':
                    self.scaler[f'input_{self.iter}'] = MinMaxScaler()
                elif scaler == 'StandardScaler':
                    self.scaler[f'input_{self.iter}'] = StandardScaler()
                self.scaler[f'input_{self.iter}'].fit(x_array)
                x_array = self.scaler[f'input_{self.iter}'].transform(x_array)
                self.X[f'input_{self.iter}']['data'] = x_array.reshape(shape_x)

        if 'classification' in self.tags.values():
            self.y_cls = y_array.astype('int')
            del y_array

        pass

    def classification(self, **options) -> np.ndarray:

        one_hot_encoding: bool = options['one_hot_encoding']

        y_array = self.y_cls
        self.classes_names[f'output_{self.iter}'] = [folder for folder in sorted(os.listdir(self.file_folder))]
        self.num_classes[f'output_{self.iter}'] = len(np.unique(y_array, axis=0))
        self.task_type[f'output_{self.iter}'] = 'classification'
        if one_hot_encoding:
            y_array = utils.to_categorical(y_array, len(np.unique(y_array)))
            self.one_hot_encoding[f'output_{self.iter}'] = True
        else:
            self.one_hot_encoding[f'output_{self.iter}'] = False

        return y_array

    def text_segmentation(self, **options) -> np.ndarray:

        def get_ohe_samples(list_of_txt, tags_index):
            tags01 = []
            indexes = []
            for txt in list_of_txt:
                tag_place = [0 for _ in range(self.num_classes[f'output_{self.iter}'])]
                for ex in txt:
                    if ex in tags_index:
                        place = np.argwhere(tags_index == ex)
                        if len(place) != 0:
                            if place[0][0] < len(open_tags.split(' ')):
                                tag_place[place[0][0]] = 1
                            else:
                                tag_place[place[0][0] - len(open_tags.split(' '))] = 0
                    else:
                        tags01.append(tag_place.copy())
                        indexes.append(ex)

            return indexes, tags01

        def get_set_from_indexes(word_indexes, length, seq_step):

            sample = []
            words_len = len(word_indexes)

            index = 0
            while index + length <= words_len:
                sample.append(word_indexes[index:index + length])
                index += seq_step

            return sample

        open_tags: str = options['open_tags']
        close_tags: str = options['close_tags']

        y_array = np.array([])

        self.num_classes[f'output_{self.iter}'] = len(open_tags.split(' '))
        self.classes_names[f'output_{self.iter}'] = open_tags.split(' ')
        self.one_hot_encoding[f'output_{self.iter}'] = True
        self.task_type[f'output_{self.iter}'] = 'segmentation'
        tags = open_tags.split(' ') + close_tags.split(' ')

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'text':
                x_len = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['x_len']
                step = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['step']
                tags_indexes = np.array([self.tokenizer[f'input_{i + 1}'].word_index[k] for k in tags])

                _, y_data = get_ohe_samples(self.sequences, tags_indexes)
                y_array = get_set_from_indexes(y_data, x_len, step)
                y_array = np.array(y_array)

                break

        return y_array

    def segmentation(self, **options) -> np.ndarray:

        def cluster_to_ohe(mask_image):

            mask_image = mask_image.reshape(-1, 3)
            km = KMeans(n_clusters=num_classes)
            km.fit(mask_image)
            labels = km.labels_
            cl_cent = km.cluster_centers_.astype('uint8')[:max(labels) + 1]
            cl_mask = utils.to_categorical(labels, max(labels) + 1)
            cl_mask = cl_mask.reshape(shape[0], shape[1], cl_mask.shape[-1])

            mask_ohe = np.zeros(shape)
            for k, (name, rgb) in enumerate(classes_dict.items()):
                mask = np.zeros(shape)

                for j, cl_rgb in enumerate(cl_cent):
                    if rgb[0] in range(cl_rgb[0] - mask_range, cl_rgb[0] + mask_range) and \
                            rgb[1] in range(cl_rgb[1] - mask_range, cl_rgb[1] + mask_range) and \
                            rgb[2] in range(cl_rgb[2] - mask_range, cl_rgb[2] + mask_range):
                        mask = cl_mask[:, :, j]

                if k == 0:
                    mask_ohe = mask
                else:
                    mask_ohe = np.dstack((mask_ohe, mask))

            return mask_ohe

        folder_name: str = options['folder_name']
        mask_range: int = options['mask_range']
        classes_names: list = options['classes_names']
        classes_colors: list = options['classes_colors']

        self.classes_names[f'output_{self.iter}'] = classes_names
        self.classes_colors[f'output_{self.iter}'] = classes_colors
        num_classes = len(classes_names)
        self.num_classes[f'output_{self.iter}'] = num_classes
        self.one_hot_encoding[f'output_{self.iter}'] = True
        self.task_type[f'output_{self.iter}'] = 'segmentation'
        classes_dict = {}
        for i in range(len(classes_names)):
            classes_dict[classes_names[i]] = classes_colors[i]

        if folder_name == '':
            folder_name = self.file_folder
        else:
            folder_name = os.path.join(self.file_folder, folder_name)

        y_array = []
        shape: tuple = ()
        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'images':
                height = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['height']
                width = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['width']
                shape = (height, width)
                break

        for _, dir_names, filename in sorted(os.walk(folder_name)):

            folders = sorted(dir_names)
            folders_num = len(dir_names) if len(dir_names) != 0 else 1
            for i in range(folders_num):
                temp_path = folder_name
                try:
                    temp_path = os.path.join(folder_name, folders[i])
                except IndexError:
                    pass

                files = sorted(os.listdir(temp_path))

                progress_bar = tqdm(files, ncols=800)
                progress_bar.set_description(f'Сохранение масок сегментации')
                idx = 0
                for file in progress_bar:
                    img = load_img(os.path.join(temp_path, file), target_size=shape)
                    image = img_to_array(img).astype('uint8')
                    image_ohe = cluster_to_ohe(image)
                    y_array.append(image_ohe)
                    if self.django_flag:
                        idx += 1
                        progress_bar_status =\
                            (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                             f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)
            break

        y_array = np.array(y_array)

        return y_array

    def object_detection(self):

        def make_y(real_boxes, classes_num):

            anchors = np.array(
                [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
            num_layers = 3
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

            real_boxes = np.array(real_boxes, dtype='float32')
            input_shape = np.array((height, width), dtype='int32')

            boxes_wh = real_boxes[..., 2:4] * input_shape

            cells = [13, 26, 52]
            y_true = [np.zeros((cells[n], cells[n], len(anchor_mask[n]), 5 + classes_num), dtype='float32') for n in
                      range(num_layers)]
            box_area = boxes_wh[:, 0] * boxes_wh[:, 1]

            anchor_area = anchors[:, 0] * anchors[:, 1]
            for r in range(len(real_boxes)):
                correct_anchors = []
                for anchor in anchors:
                    correct_anchors.append([min(anchor[0], boxes_wh[r][0]), min(anchor[1], boxes_wh[r][1])])
                correct_anchors = np.array(correct_anchors)
                correct_anchors_area = correct_anchors[:, 0] * correct_anchors[:, 1]
                iou = correct_anchors_area / (box_area[r] + anchor_area - correct_anchors_area)
                best_anchor = np.argmax(iou, axis=-1)

                for m in range(num_layers):
                    if best_anchor in anchor_mask[m]:
                        h = np.floor(real_boxes[r, 0] * cells[m]).astype('int32')
                        j = np.floor(real_boxes[r, 1] * cells[m]).astype('int32')
                        k = anchor_mask[m].index(int(best_anchor))
                        c = real_boxes[r, 4].astype('int32')
                        y_true[m][j, h, k, 0:4] = real_boxes[r, 0:4]
                        y_true[m][j, h, k, 4] = 1
                        y_true[m][j, h, k, 5 + c] = 1
                        break

            return y_true[0], y_true[1], y_true[2]

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'images':
                height = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['height']
                width = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['width']
                break

        data = {}
        class_names = {}

        # obj.data
        with open(os.path.join(self.file_folder, 'obj.data'), 'r') as dt:
            d = dt.read()
        for elem in d.split('\n'):
            if elem:
                elem = elem.split(' = ')
                data[elem[0]] = elem[1]
        num_classes = int(data['classes'])
        self.num_classes[f'output_{self.iter}'] = num_classes
        self.task_type[f'output_{self.iter}'] = 'object_detection'
        self.task_type[f'output_{self.iter+1}'] = 'object_detection'
        self.task_type[f'output_{self.iter+2}'] = 'object_detection'

        # obj.names
        with open(os.path.join(self.file_folder, data["names"].split("/")[-1]), 'r') as dt:
            names = dt.read()
        for i, elem in enumerate(names.split('\n')):
            if elem:
                class_names[i] = elem

        # list of txt
        txt_list = []
        with open(os.path.join(self.file_folder, data["train"].split("/")[-1]), 'r') as dt:
            images = dt.read()
        for elem in images.split('\n'):
            if elem:
                idx = elem.rfind('.')
                elem = elem.replace(elem[idx:], '.txt')
                txt_list.append(os.path.join(*elem.split('/')[1:]))

        input_1 = []
        input_2 = []
        input_3 = []

        # read txt files + make y's
        for txt_file in sorted(txt_list):
            with open(os.path.join(self.file_folder, txt_file), 'r') as txt:
                bb_file = txt.read()
            real_bb = []
            for elem in bb_file.split('\n'):
                tmp = []
                if elem:
                    for num in elem.split(' '):
                        tmp.append(float(num))
                    real_bb.append(tmp)
            real_bb = np.array(real_bb)
            real_bb = real_bb[:, [1, 2, 3, 4, 0]]
            out1, out2, out3 = make_y(real_boxes=real_bb, classes_num=num_classes)
            input_1.append(out1)
            input_2.append(out2)
            input_3.append(out3)

        input_1 = np.array(input_1)
        input_2 = np.array(input_2)
        input_3 = np.array(input_3)

        return input_1, input_2, input_3

    def prepare_user_dataset(self, dataset_dict: dict, is_save=True):

        cur_time = time()

        self.name = dataset_dict['parameters']['name']
        self.user_tags = dataset_dict['parameters']['user_tags'].split(' ')
        self.divide_ratio = [dataset_dict['parameters']['train_part'], dataset_dict['parameters']['val_part'],
                             dataset_dict['parameters']['test_part']]

        self.user_parameters['inp'] = dataset_dict['inputs']
        self.user_parameters['out'] = dataset_dict['outputs']

        for i in range(len(self.user_parameters['inp'])):
            self.tags[f'input_{i + 1}'] = dataset_dict['inputs'][f'input_{i + 1}']['tag']
        for i in range(len(self.user_parameters['out'])):
            self.tags[f'output_{i + 1}'] = dataset_dict['outputs'][f'output_{i + 1}']['tag']

        self.iter = 0
        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'audio':
                getattr(self, 'audio')(**self.user_parameters['inp'][f'input_{i + 1}']['parameters'])
            else:
                self.iter += 1
                self.X[f'input_{i + 1}'] = {'data_name': self.user_parameters['inp'][f'input_{i + 1}']['name'],
                                            'data': getattr(self, self.user_parameters['inp'][f'input_{i + 1}']['tag'])(
                                                **self.user_parameters['inp'][f'input_{i + 1}']['parameters'])}

        for i in range(len(self.user_parameters['out'])):
            self.iter = i + 1
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'object_detection':
                outputs = getattr(self, self.user_parameters['out'][f'output_{i + 1}']['tag'])(
                    **self.user_parameters['out'][f'output_{i + 1}']['parameters'])
                for k in range(3):
                    self.Y[f'output_{i + k + 1}'] = {
                        'data_name': self.user_parameters['out'][f'output_{i + 1}']['name'], 'data': outputs[k]}
                    self.tags[f'output_{i + k + 1}'] = dataset_dict['outputs'][f'output_{i + 1}']['tag']
            else:
                self.Y[f'output_{i + 1}'] = {'data_name': self.user_parameters['out'][f'output_{i + 1}']['name'],
                                             'data': getattr(self,
                                                             self.user_parameters['out'][f'output_{i + 1}']['tag'])(
                                                 **self.user_parameters['out'][f'output_{i + 1}']['parameters'])}

        train_mask = []
        val_mask = []
        test_mask = []

        for i in range(len(self.peg) - 1):
            indices = np.arange(self.peg[i], self.peg[i + 1])
            train_len = int(self.divide_ratio[0] * len(indices))
            val_len = int(self.divide_ratio[1] * len(indices))
            indices = indices.tolist()
            train_mask.extend(indices[:train_len])
            val_mask.extend(indices[train_len:train_len + val_len])
            test_mask.extend(indices[train_len + val_len:])

        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'timeseries':
                length = self.user_parameters['out'][f'output_{i + 1}']['parameters']['length']
                train_mask = train_mask[:-length]
                val_mask = val_mask[:-length]

        if not dataset_dict['parameters']['preserve_sequence']:
            random.shuffle(train_mask)
            random.shuffle(val_mask)
            random.shuffle(test_mask)

        for inp in self.X.keys():
            self.X[inp]['data'] = (self.X[inp]['data'][train_mask], self.X[inp]['data'][val_mask],
                                   self.X[inp]['data'][test_mask])
        for out in self.Y.keys():
            self.Y[out]['data'] = (self.Y[out]['data'][train_mask], self.Y[out]['data'][val_mask],
                                   self.Y[out]['data'][test_mask])

        inp_datatype = []
        for inp in self.X.keys():
            self.input_shape[inp] = self.X[inp]['data'][0].shape[1:]
            inp_datatype.append(self._set_datatype(shape=self.X[inp]['data'][0].shape))

        for out in self.Y.keys():
            self.output_shape[out] = self.Y[out]['data'][0].shape[1:]
            self.output_datatype[out] = self._set_datatype(shape=self.Y[out]['data'][0].shape)

        self.input_datatype = ' '.join(inp_datatype)

        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            x = ['x_train', 'x_val', 'x_test']
            y = ['y_train', 'y_val', 'y_test']
            for inp in self.X.keys():
                for i, item in enumerate(self.X[inp]['data']):
                    if isinstance(item, np.ndarray):
                        print(f'Размерность {inp} - {x[i]}: {self.X[inp]["data"][i].shape}')
            for out in self.Y.keys():
                for i, item in enumerate(self.Y[out]['data']):
                    if isinstance(item, np.ndarray):
                        print(f'Размерность {out} - {y[i]}: {self.Y[out]["data"][i].shape}')

        # temp_attributes = ['iter', 'sequences', 'y_cls']  # 'df' , 'peg'
        # for item in temp_attributes:
        #     if hasattr(self, item):
        #         delattr(self, item)

        self.dts_prepared = True
        if is_save:
            print('Идёт сохранение датасета.')
            os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}'), exist_ok=True)
            if self.X:
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'arrays'), exist_ok=True)
            if self.scaler:
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'scalers'), exist_ok=True)
            if self.tokenizer:
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer'), exist_ok=True)
            if self.word2vec:
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec'), exist_ok=True)
            if self.tsgenerator:
                os.makedirs(os.path.join(self.trds_path, f'dataset {self.name}', 'tsgenerator'), exist_ok=True)

            for arr in self.X.keys():
                if self.X[arr]:
                    joblib.dump(self.X[arr],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', f'{arr}.gz'))
            for arr in self.Y.keys():
                if self.Y[arr]:
                    joblib.dump(self.Y[arr],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'arrays', f'{arr}.gz'))
            for scaler in self.scaler.keys():
                if self.scaler[scaler]:
                    joblib.dump(self.scaler[scaler],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'scalers', f'{scaler}.gz'))
            for tok in self.tokenizer.keys():
                if self.tokenizer[tok]:
                    joblib.dump(self.tokenizer[tok],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'tokenizer', f'{tok}.gz'))
            for w2v in self.word2vec.keys():
                if self.word2vec[w2v]:
                    joblib.dump(self.word2vec[w2v],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'word2vec', f'{w2v}.gz'))
            for tsg in self.tsgenerator.keys():
                if self.tsgenerator[tsg]:
                    joblib.dump(self.tsgenerator[tsg],
                                os.path.join(self.trds_path, f'dataset {self.name}', 'tsgenerator', f'{tsg}.gz'))

            data = {}
            attributes = ['name', 'source', 'tags', 'user_tags', 'classes_colors', 'classes_names', 'dts_prepared',
                          'language', 'num_classes', 'one_hot_encoding', 'task_type']
            for attr in attributes:
                data[attr] = self.__dict__[attr]
            data['date'] = datetime.now().astimezone(timezone('Europe/Moscow')).isoformat()
            data['size'] = self._get_size(os.path.join(self.trds_path, f'dataset {self.name}'))
            with open(os.path.join(self.trds_path, f'dataset {self.name}', 'config.json'), 'w') as fp:
                json.dump(data, fp)
            print(f'Файлы датасета сохранены в папку {os.path.join(self.trds_path, f"dataset {self.name}")}')
            print(f'Json сохранен в файл {os.path.join(self.trds_path, f"dataset {self.name}", "config.json")}')

        self.Exch.reset_stop_flag()

        return self
