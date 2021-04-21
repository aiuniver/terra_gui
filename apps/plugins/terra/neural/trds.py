from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from time import time
from time import sleep
from PIL import Image
from inspect import getmembers, signature
import subprocess
from subprocess import STDOUT, check_call
import librosa
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import gdown
import zipfile
import re
import pymorphy2
from shutil import rmtree
from gensim.models import word2vec
from tqdm.notebook import tqdm
import threading
from io import open as ioopen
from IPython.display import clear_output
from apps.plugins.terra import colab_exchange

__version__ = 0.223


class DTS(object):

    def __init__(self, exch_obj=colab_exchange):

        # if 'custom' in globals().keys():
        #     for key, value in custom.__dict__.items():
        #         self.__dict__[key] = value

        self.Exch = exch_obj
        self.django_flag = True
        self.divide_ratio = [(0.8, 0.2), (0.8, 0.1, 0.1)]
        self.file_folder: str = ''
        self.name: str = ''
        self.source: str = ''
        self.tags: list = []
        self.source_datatype: list = []
        self.source_shape: list = []
        self.input_datatype: list = []
        self.input_shape: list = []
        self.num_classes: int = 0
        self.classes_names: list = []
        self.classes_colors: list = []
        self.language: str = ''
        self.dts_prepared: bool = False

        self.X: dict = {}
        self.Y: dict = {}

        pass

    def list_data(self):

        data = {
            'трафик': 'Трафик сайта компании.',
            'трейдинг': 'Курсы акций.',
            'умный_дом': 'Голосовые команды для управления умным домом.',
            'квартиры': 'База квартир в Москве.',
            'диалоги': 'Диалоги в формате вопрос-ответ.',
            'автомобили': 'Классификация автомобилей на 2 класса.',
            'автомобили_3': 'Классификация автомобилей на 3 класса.',
            'заболевания': 'Классификация болезней по симптомам.',
            'договоры': 'Сегментация договоров.',
            'самолеты': 'Сегментация самолетов.',
            'болезни': 'Классификация кожных заболеваний по фотографиям.'
        }

        return pd.DataFrame(data.items()).rename(columns={0: "Название", 1: "Описание"})

    def list_functions(self) -> pd.DataFrame:
        """

        Returns:  table in pandas format with two columns: function name and description

        """
        data = {
            'keras_datasets(dataset, **options)': 'Загрузка стандартных баз: mnist, fashion_mnist, cifar10, cifar100, imdb, '
                                                  'reuters_newswire, boston_housing',
            'image_classification(shape, folder_name=None, **options)': 'Создание массивов для задачи классификации изображений.',
            'image_segmentation(shape, classes_dict, mask_range, folder_name=None, **options)': 'Создание массивов для задачи сегментации изображений.',
            'text_classification(max_words_count, x_len, step, folder_name=None, **options)': 'Создание массивов для задачи классификации текста.',
            'text_segmentation(max_words_count, x_len, step, embedding_size, num_classes, folder_name=None)': 'Создание '
                                                                                                              'массивов ''для задачи '
                                                                                                              'сегментации текста. ',
            'voice_recognition(sample_rate, length, folder_name=None, **options)': 'Создание массивов для распознавания звуковых файлов.',
            'data_regression(filename, x_len, val_len, **options)': 'Создание массивов для задачи регрессии.',
            'as_array()': 'Возвращает массивы X Y'
        }
        pd.set_option('display.max_colwidth', None)

        return pd.DataFrame(data.items()).rename(columns={0: "Название", 1: "Описание"})

    def _set_tag(self, name):

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

    def _set_language(self, name):

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

    def _set_source(self, name):

        source = {'mnist': 'tensorflow.keras',
                  'fashion_mnist': 'tensorflow.keras',
                  'cifar10': 'tensorflow.keras',
                  'cifar100': 'tensorflow.keras',
                  'imdb': 'tensorflow.keras',
                  'boston_housing': 'tensorflow.keras',
                  'reuters': 'tensorflow.keras',
                  'sber': 'terra_ai',
                  'автомобили': 'terra_ai',
                  'автомобили_3': 'terra_ai',
                  'самолеты': 'terra_ai',
                  'умный_дом': 'terra_ai',
                  'договоры': 'terra_ai',
                  'трейдинг': 'terra_ai',
                  'квартиры': 'terra_ai',
                  'болезни': 'terra_ai',
                  'заболевания': 'terra_ai',
                  'губы': 'terra_ai',
                  'жанры_музыки': 'terra_ai'
                  }

        if name in source.keys():
            return source[name]
        else:
            return 'custom_dataset'

    def _set_datatype(self, **kwargs):

        dtype = {1: 'DIM',
                 2: 'DIM',
                 3: '1D',
                 4: '2D',
                 5: '3D'
                 }

        if 'shape' in kwargs.keys():
            return dtype[len(kwargs['shape'])]
        elif 'text' in kwargs.keys() and kwargs['text'] == True:
            return 'Text'

    def load_data(self, name, link=None, **options) -> None:

        """
        Create folder and download base in it. Does not change the files original format.
        If base is on the standard base list also print detailed information about specified base.

        Examples:
            trds.DataLoader().load_data('договоры');

            trds.DataLoader().load_data('base_name', url)
        Args:
            name (str): name of the base for downloading;

            link (str): url where base is located
        """

        self.name = name
        data = {
            'трафик': ['traff.csv'],
            'трейдинг': ['shares.zip'],
            'автомобили': ['car_2.zip'],
            'умный_дом': ['cHome.zip'],
            'квартиры': ['moscow.csv'],
            'диалоги': ['dialog.txt'],
            'автомобили_3': ['car.zip'],
            'заболевания': ['symptoms.zip'],
            # 'люди':['master.zip', 'coco2017val.zip', 'coco128.zip'], # Этой базы нет в aiu_bucket
            'договоры': ['docs.zip'],
            'самолеты': ['airplane.zip', 'segment.zip'],
            'болезни': ['origin.zip', 'segmentation.zip'],
            'губы': ['lips.zip'],
            'жанры_музыки': ['genres.zip'],
            'sber': ['SBER_MIN60.txt']
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
            # 'люди':['master.zip', 'coco2017val.zip', 'coco128.zip'], # Этой базы нет в aiu_bucket
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

        default_path = pathlib.Path().absolute()
        working_path = default_path.joinpath(os.path.join('datasets', 'sources'))
        if link:
            if 'drive.google' in link:
                filename = name
                name = name.split('.')[0]
                file_id = max(link.split('/'), key=len)
            else:
                filename = link.split('/')[-1]
            if 'save_path' in options.keys():
                main_folder = pathlib.Path(options['save_path'])
            else:
                main_folder = working_path
            file_folder = pathlib.Path(os.path.join(main_folder, name))
            if not file_folder.exists():
                os.makedirs(file_folder)
            if 'zip' in filename or 'zip' in link:
                file_path = pathlib.Path(os.path.join(main_folder, name, 'tmp', filename))
                temp_folder = os.path.join(file_folder, 'tmp')
                os.mkdir(temp_folder)
                os.chdir(temp_folder)
                if 'drive.google' in link:
                    gdown.download('https://drive.google.com/uc?id=' + file_id, filename, quiet=self.django_flag)
                else:
                    gdown.download(link, filename, quiet=self.django_flag)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(file_folder)
                    os.chdir(str(default_path))
                rmtree(temp_folder, ignore_errors=True)
            else:
                os.chdir(file_folder)
                if 'drive.google' in link:
                    gdown.download('https://drive.google.com/uc?id=' + file_id, filename, quiet=self.django_flag)
                else:
                    gdown.download(link, filename, quiet=self.django_flag)
                os.chdir(str(default_path))
        else:
            if name in data.keys():
                self.tags = self._set_tag(self.name)
                self.language = self._set_language(self.name)
                for base in data[name]:
                    if 'save_path' in options.keys():
                        main_folder = pathlib.Path(options['save_path'])
                    else:
                        main_folder = working_path
                    file_folder = main_folder.joinpath(name)
                    if not file_folder.exists():
                        os.makedirs(file_folder)
                    url = 'https://storage.googleapis.com/terra_ai/DataSets/' + base
                    if 'zip' in base:
                        file_path = pathlib.Path(os.path.join(main_folder, name, 'tmp', base))
                        temp_folder = file_folder.joinpath('tmp')
                        os.mkdir(temp_folder)
                        os.chdir(temp_folder)
                        gdown.download(url, base, quiet=self.django_flag)
                        os.chdir(str(default_path))
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(file_folder)
                            os.chdir(str(default_path))
                        rmtree(temp_folder, ignore_errors=True)
                    else:
                        os.chdir(file_folder)
                        gdown.download(url, base, quiet=self.django_flag)
                        os.chdir(str(default_path))
                if not self.django_flag:
                    if name in reference.keys():
                        print(reference[name])
            else:
                if not name in data.keys():
                    if self.django_flag:
                        self.Exch.print_error(('Error', 'Данной базы нет в списке готовых баз.'))
                    else:
                        assert name in data.keys(), 'Данной базы нет в списке готовых баз.'
        self.file_folder = str(file_folder)
        self.source = self._set_source(name)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        return self

    def keras_datasets(self, dataset, **options):
        """
        Prepare data for processing in neural network an shows examples from  chosen dataset

        Example:
            mnist = trds.DTS()
            mnist.keras_datasets(mnist, net = 'conv', scale = True)

        Args:
            dataset (str):  name of dataset from keras.datasets (mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing).

        **options: Additional parameters (used for processing images);
            net = 'linear', - flatten by multiplying shapes fitting it  for  Dense layer;
            net = 'conv', - adds dimension to fit it for convolution layer;
            scaler = 'MinMaxScaler', - apply MinMaxScaler (range 0 to 1)
            scaler = 'StandardScaler', - apply StandardScaler (range -1 to 1)
            test = True, - return test array.

        Returns:
             Arrays prepared for processing in neural network.
        """

        def print_data(name, x_train, y_train):
            pics = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
            text = ['imdb', 'reuters', 'boston_housing']

            if name in pics:
                fig, axs = plt.subplots(1, 10, figsize=(25, 3))
                for i in range(10):
                    label_indexes = np.where(y_train == i)[0]
                    index = random.choice(label_indexes)
                    img = x_train[index]
                    title = y_train[index]
                    if name in ['mnist', 'fashion_mnist']:
                        axs[i].imshow(Image.fromarray(img), cmap='gray')
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names[title]}')
                    else:
                        axs[i].imshow(Image.fromarray(img))
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names[title[0]]}')

            if name in text:
                if name in ['imdb', 'reuters']:
                    pd.DataFrame({'x_train': x_train, 'y_train': y_train}).head()
                else:
                    df = pd.DataFrame(x_train)
                    df['y_train'] = y_train
                    df.head()

            pass

        cur_time = time()
        self.name = dataset.lower()
        self.tags = self._set_tag(self.name)
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
        if not dataset.lower() in data.keys():
            self.Exch.print_error(('Error', 'Данного датасета нет в списке стандартных датасетов keras.'))
            if not self.django_flag:
                assert dataset.lower() in data.keys(), 'Данного датасета нет в списке стандартных датасетов keras.'
        progress_bar = tqdm(range(1), ncols=800)
        progress_bar.set_description(f'Загрузка датасета {self.name}')
        idx = 0
        for _ in progress_bar:
            (x_Train, y_Train), (x_Val, y_Val) = data[self.name].load_data()
            if self.django_flag:
                idx += 1
                progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                       f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)

        self.source_shape = x_Train.shape if len(x_Train.shape) < 2 else x_Train.shape[1:]
        self.language = self._set_language(self.name)
        self.source_datatype = self._set_datatype(shape=x_Train.shape)
        if 'classification' in self.tags:
            self.num_classes = len(np.unique(y_Train, axis=0))
            if self.name == 'fashion_mnist':
                self.classes_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                      'Sneaker', 'Bag', 'Ankle boot']
            elif self.name == 'cifar10':
                self.classes_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                      'truck']
            else:
                self.classes_names = [str(i) for i in range(self.num_classes)]

        if not self.django_flag:
            print_data(self.name, x_Train, y_Train)

        if 'net' in options.keys() and self.name in list(data.keys())[:4]:
            if options['net'].lower() == 'linear':
                x_Train = x_Train.reshape((-1, np.prod(np.array(x_Train.shape)[1:])))
                x_Val = x_Val.reshape((-1, np.prod(np.array(x_Val.shape)[1:])))
            elif options['net'].lower() == 'conv':
                if len(x_Train.shape) == 3:
                    x_Train = x_Train[..., None]
                    x_Val = x_Val[..., None]

        self.input_shape = x_Train.shape if len(x_Train.shape) < 2 else x_Train.shape[1:]
        self.input_datatype = self._set_datatype(shape=x_Train.shape)

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

            if self.name == 'imdb' or self.name == 'reuters':
                if not self.django_flag:
                    print(
                        f'Scaling required dataset is currently unavaliable. {options["scaler"]} was not implemented.')
            else:
                shape_xt = x_Train.shape
                shape_xv = x_Val.shape
                x_Train = x_Train.reshape(-1, 1)
                x_Val = x_Val.reshape(-1, 1)

                if 'classification' not in self.tags:
                    shape_yt = y_Train.shape
                    shape_yv = y_Val.shape
                    y_Train = y_Train.reshape(-1, 1)
                    y_Val = y_Val.reshape(-1, 1)

                if options['scaler'] == 'MinMaxScaler':
                    self.x_Scaler = MinMaxScaler()
                    if 'classification' not in self.tags:
                        self.y_Scaler = MinMaxScaler()

                elif options['scaler'] == 'StandardScaler':
                    self.x_Scaler = StandardScaler()
                    if 'classification' not in self.tags:
                        self.y_Scaler = StandardScaler()

                self.x_Scaler.fit(x_Train)
                x_Train = self.x_Scaler.transform(x_Train)
                x_Val = self.x_Scaler.transform(x_Val)
                x_Train = x_Train.reshape(shape_xt)
                x_Val = x_Val.reshape(shape_xv)
                if 'classification' not in self.tags:
                    self.y_Scaler.fit(y_Train)
                    y_Train = self.y_Scaler.transform(y_Train)
                    y_Val = self.y_Scaler.transform(y_Val)
                    y_Train = y_Train.reshape(shape_yt)
                    y_Val = y_Val.reshape(shape_yv)

        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] == True:
            if 'classification' in self.tags:
                y_Train = utils.to_categorical(y_Train, len(np.unique(y_Train, axis=0)))
                y_Val = utils.to_categorical(y_Val, len(np.unique(y_Val, axis=0)))
            else:
                if not self.django_flag:
                    print(f'One-Hot encoding only available for classification which {self.name} was not meant for. '
                          f'One-Hot encoding was not implemented.')

        self.X['input_1'] = (x_Train, x_Val, None)
        self.Y['output_1'] = (y_Train, y_Val, None)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            x_Val, x_Test, y_Val, y_Test = train_test_split(x_Val, y_Val, test_size=1 - split_size, shuffle=True)
            self.X['input_1'] = (x_Train, x_Val, x_Test)
            self.Y['output_1'] = (y_Train, y_Val, y_Test)

        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            x_arrays = ['x_train', 'x_val', 'x_test']
            for i, item_x in enumerate(self.X['input_1']):
                if item_x is not None:
                    print(f"Размерность {x_arrays[i]}: {item_x.shape}")
            y_arrays = ['y_train', 'y_val', 'y_test']
            for i, item_y in enumerate(self.Y['output_1']):
                if item_y is not None:
                    print(f"Размерность {y_arrays[i]}: {item_y.shape}")

        return self

    def image_classification(self, shape, folder_name=None, **options):
        """
        Prepare data for processing in neural network for image classification. Show examples from  chosen dataset.

        Example:
            classification = trds.DTS()
            classification.image_classification(shape=(96, 128), split_size=0.2, scale=True, one_hot_encoding = True)

        Args:
            shape (tuple): necessary shape of image  (width, length);
            folder_name (str): path to folder with images.

         **options:
            scaler = 'MinMaxScaler':  normalizing data using MinMaxScaler;
            scaler = 'StandardScaler': normalizing data using StandardScaler;
            one_hot_encoding = True: transform Y to OHE format.

        Returns:
               Arrays prepared for processing in neural network.
        """

        def load_image(img_path, shape):
            """

            Args:
                img_path (str): path to image
                shape (tuple): required shape of downloading image (width,legth)

            Returns:
                 numpy array

            """

            img = load_img(img_path, target_size=shape)
            array = img_to_array(img)

            return array.astype('uint8')

        cur_time = time()
        if folder_name == None:
            folder_name = self.file_folder
        X = []
        Y = np.array([]).astype('uint8')
        folders = sorted(os.listdir(folder_name))
        for k, folder in enumerate(folders, 1):
            i = folders.index(folder)
            files = [f for f in sorted(os.listdir(os.path.join(folder_name, folder)))]
            progress_bar = tqdm(files, ncols=800)
            progress_bar.set_description(f'Сохранение изображений из папки {folder}')
            if len(self.source_shape) < 3:
                source_shape = Image.open(os.path.join(folder_name, folder, files[0])).size
                self.source_shape = (source_shape[1], source_shape[0], 3)
            idx = 1
            for file in progress_bar:
                X.append(load_image(os.path.join(folder_name, folder, file), shape))
                Y = np.append(Y, i)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total and k == len(folders):
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)
        X = np.array(X)
        self.source_datatype = self._set_datatype(shape=X.shape)
        count = len(np.unique(Y, axis=0))
        self.num_classes = count
        if self.name == 'автомобили':
            self.classes_names = ['Ferrari', 'Mercedes']
        elif self.name == 'автомобили_3':
            self.classes_names = ['Ferrari', 'Mercedes', 'Renault']
        else:
            self.classes_names = [str(i) for i in range(count)]

        if not self.django_flag:
            fig, ax = plt.subplots(1, count, figsize=(count * 3, 6))
            for i in range(count):
                index = np.where(Y == i)[0]
                index = np.random.choice(index, 1)[0]
                ax[i].imshow(X[index])
                ax[i].set_title(f'{i}: {self.classes_names[i]}')

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

            shape_x = X.shape
            X = X.reshape(-1, 1)

            if options['scaler'] == 'MinMaxScaler':
                self.x_Scaler = MinMaxScaler()

            elif options['scaler'] == 'StandardScaler':
                self.x_Scaler = StandardScaler()

            self.x_Scaler.fit(X)
            X = self.x_Scaler.transform(X)
            X = X.reshape(shape_x)

        if 'net' in options.keys() and options['net'].lower() == 'linear':
            X = X.reshape(-1, np.prod(np.array(X.shape)[1:]))

        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] == True:
            Y = utils.to_categorical(Y, len(np.unique(Y, axis=0)))

        self.input_shape = X.shape[1:]
        self.input_datatype = self._set_datatype(shape=X.shape)
        self.x_Train, self.x_Val, self.y_Train, self.y_Val = train_test_split(X, Y, test_size=self.divide_ratio[0][1],
                                                                              shuffle=True)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size,
                                                                                shuffle=True)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def image_segmentation(self, shape, classes_dict, mask_range=50, folder_name=None, **options):

        """
        Prepare data for processing in neural network for image segmentation. Show examples from chosen dataset_name.

        Example:
            segmentation = trds.DTS()
            segmentation.image_segmentation(shape=(96, 128), split_size=0.2, num_classes=2)

        Args:
            shape (tuple):  necessary image shape (width, length);
            num_classes (int):  number of classes for segmentation;
            folder_name (str): name of folder with images.

        **options:
            mask_colors = [[0,0,0], ...] list of colors in segmentation masks;
            mask_classes = [0, ...] list of color classes in segmentation masks;
            mask_range = 50 range for RGB channels
            scale = True normalizing data by division on 255.

        Returns:
            Array prepared for processing in neural network.
        """

        def load_image(img_path, shape):
            """

            Args:
                img_path (str): path to image
                shape (tuple): required shape of downloading image (width,legth)

            Returns:
                 numpy array

            """

            img = load_img(img_path, target_size=shape)
            array = img_to_array(img)

            return array.astype('uint8')

        def cluster_to_ohe(image):
            """
            Args:
                image (array):

            Returns:
                OHE type array
            """

            image = image.reshape(-1, 3)
            km = KMeans(n_clusters=self.num_classes)
            km.fit(image)
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

        cur_time = time()
        self.classes_names = list(classes_dict.keys())
        self.classes_colors = list(classes_dict.values())
        self.num_classes = len(self.classes_names)
        if folder_name == None:
            folder_name = self.file_folder
        X = []
        Y = []
        folders = sorted(os.listdir(folder_name))
        for i, folder in enumerate(folders):
            if i == 0:
                progress_bar = tqdm(sorted(os.listdir(os.path.join(folder_name, folder))), ncols=800)
                progress_bar.set_description(f'Сохранение изображений из папки {folder}')
                idx = 0
                for j, file in enumerate(progress_bar):
                    if len(self.source_shape) < 3:
                        source_shape = Image.open(os.path.join(folder_name, folder, file)).size
                        self.source_shape = (source_shape[1], source_shape[0], 3)
                    X.append(load_image(os.path.join(folder_name, folder, file), shape))
                    if self.django_flag:
                        idx += 1
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)
                X = np.array(X)
                if not self.django_flag:
                    index = np.random.choice(X.shape[0], 1)[0]
                    plt.figure(figsize=(6, 3))
                    plt.imshow(X[index])
                    plt.show()
            if i == 1:
                if not self.django_flag:
                    fig, ax = plt.subplots(1, len(classes_dict.keys()), figsize=(2 * self.num_classes, 3))
                    for i, (key, value) in enumerate(classes_dict.items()):
                        ax[i].imshow(np.full((30, 30, 3), value))
                        ax[i].set_title(key)
                        ax[i].axis('off')
                    plt.show()

                progress_bar = tqdm(sorted(os.listdir(os.path.join(folder_name, folder))), ncols=800)
                progress_bar.set_description(f'Сохранение изображений из папки {folder}')
                idx = 0
                for file in progress_bar:
                    image = load_image(os.path.join(folder_name, folder, file), shape)
                    image_ohe = cluster_to_ohe(image)
                    Y.append(image_ohe)
                    if self.django_flag:
                        idx += 1
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total:
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)
                Y = np.array(Y)
        self.source_datatype = self._set_datatype(shape=X.shape)

        if not self.django_flag:
            fig, ax = plt.subplots(1, self.num_classes, figsize=(4 * self.num_classes, 6))
            for i in range(self.num_classes):
                ax[i].imshow(Y[index, :, :, i])
                ax[i].set_title(self.classes_names[i])
            plt.show()

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

            shape_x = X.shape
            X = X.reshape(-1, 1)

            if options['scaler'] == 'MinMaxScaler':
                self.x_Scaler = MinMaxScaler()

            elif options['scaler'] == 'StandardScaler':
                self.x_Scaler = StandardScaler()

            self.x_Scaler.fit(X)
            X = self.x_Scaler.transform(X)
            X = X.reshape(shape_x)

        self.input_shape = X.shape[1:]
        self.input_datatype = self._set_datatype(shape=X.shape)
        self.x_Train, self.x_Val, self.y_Train, self.y_Val = train_test_split(X, Y, test_size=self.divide_ratio[0][1],
                                                                              shuffle=True)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size,
                                                                                shuffle=True)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def text_classification(self, max_words_count, x_len, step, folder_name=None, **options):
        """
        Prepare data for processing in neural network for text classification analysis.

        Examples:
            txt_clsf = trds.DTS();
            txt_clsf.text_classification(max_words_count=20000, x_len=100, step=30, bag_of_words=True, one_hot_encoding=True)

        Args:
            max_words_count (int): size of vocabulary (amount of words);
            x_len (int): length of text snippet;
            step (int): how many words to shift on each step when preparing text to processing;
            folder_name (str): link to the folder.

        **options:
            bag_of_words = True: provides transformation in  Bag Of Words;
            one_hot_encoding = True: transform Y to OHE format.
            test = True: returns test array

        Returns:
            Arrays prepared for processing in neural network and tokenizer if requested.
        """

        def read_text(file_path):

            with ioopen(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
                text = text.replace('\n', ' ')

            return text

        def create_sets_multi_classes(word_indexes, x_len, step):

            def get_set_from_indexes(word_indexes, x_len, step):

                x_sample = []
                words_len = len(word_indexes)
                index = 0
                while index + x_len <= words_len:
                    x_sample.append(word_indexes[index:index + x_len])
                    index += step

                return x_sample

            classes_x_samples = []
            for w_i in word_indexes:
                classes_x_samples.append(get_set_from_indexes(w_i, x_len, step))

            x_samples = []
            y_samples = []

            progress_bar = tqdm(range(self.num_classes), ncols=800)
            idx = 0
            for t in progress_bar:
                progress_bar.set_description(f'Обработка класса {self.classes_names[t]}')
                x_t = classes_x_samples[t]
                for i in range(len(x_t)):
                    x_samples.append(x_t[i])
                if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] == True:
                    curr_y = utils.to_categorical(t, self.num_classes)
                    for _ in range(len(x_t)):
                        y_samples.append(curr_y)
                else:
                    for _ in range(len(x_t)):
                        y_samples.append(t)

                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total and idx == len(classes_x_samples):
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)

            x_samples = np.array(x_samples)
            y_samples = np.array(y_samples)

            return (x_samples, y_samples)

        cur_time = time()
        if folder_name == None:
            folder_name = self.file_folder
        self.source_datatype = self._set_datatype(text=True)
        text_list = []

        folder_list = sorted(os.listdir(folder_name))
        for k, folder in enumerate(folder_list, 1):
            progress_bar = tqdm(sorted(os.listdir(os.path.join(folder_name, folder))), ncols=800)
            progress_bar.set_description(f'Загрузка файлов из папки {folder}')
            idx = 0
            for file in progress_bar:
                self.classes_names.append(file.split('.')[0])
                text_list.append(read_text(os.path.join(folder_name, folder, file)))
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total and k == len(folder_list):
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)
        self.num_classes = len(self.classes_names)
        tokenizer = Tokenizer(num_words=max_words_count, filters='–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff',
                              lower=True, split=' ', char_level=False, oov_token='unknown')
        tokenizer.fit_on_texts(text_list)
        self.tokenizer = tokenizer
        text_indexes = tokenizer.texts_to_sequences(text_list)

        symbols_train_text = 0
        words_train_text = 0
        word_num = []
        char_num = []
        for i in range(len(self.classes_names)):
            word_num.append(len(text_indexes[i]))
            char_num.append(len(text_list[i]))
            symbols_train_text += len(text_list[i])
            words_train_text += len(text_indexes[i])
        if not self.django_flag:
            df = pd.DataFrame(
                {'Название класса': self.classes_names, 'Количество слов': word_num, 'Количество символов': char_num})
            display(df)
            print(f'В сумме {symbols_train_text} символов, {words_train_text} слов')
        X, Y = create_sets_multi_classes(text_indexes, x_len, step)
        self.source_shape = X.shape[1:]
        self.input_shape = X.shape[1:]
        self.input_datatype = self._set_datatype(shape=X.shape)
        if 'bag_of_words' in options.keys() and options['bag_of_words'] == True:
            X = tokenizer.sequences_to_matrix(X.tolist())

        self.x_Train, self.x_Val, self.y_Train, self.y_Val = train_test_split(X, Y, test_size=self.divide_ratio[0][1],
                                                                              shuffle=True)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size,
                                                                                shuffle=True)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def text_segmentation(self, max_words_count, x_len, step, embedding_size, num_classes, folder_name=None, **options):
        """
       Prepare data for processing in neural network for text  analysis

        Examples:
            txt_segm = trds.DTS()
            txt_segm.text_segmentation(max_words_count=20000, x_len=256, step=30, embedding_size=300, num_classes=6)

        Args:
            max_words_count (int): size of vocabulary (amount of words);
            x_len (int) : length of text snippet;
            step (int) : how many words to shift on each step when  preparing text to processing;
            embedding_size (int): dimension of array words are presented for processing;
            split_size (float): test to train ratio;
            num_classes (int) : indicate number of classes on which data will be separated;
            folder_name (str): name of the folder.

        **options:
            test = True, returns test array.

        Returns:
            Arrays prepared for processing in neural network.

        """

        def read_text(file_path):

            with ioopen(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
            del_symbols = ['\n', "\t", "\ufeff", ".", "_", "-", ",", "!", "?", "–", "(", ")", "«", "»", "№", ";"]
            for d_s in del_symbols:
                text = text.replace(d_s, ' ')
            text = re.sub("[.]", " ", text)
            text = re.sub(":", " ", text)
            text = re.sub("<", " <", text)
            text = re.sub(">", "> ", text)
            text = ' '.join(text.split())
            text = text.lower()

            return text

        def text_to_words(text):

            morph = pymorphy2.MorphAnalyzer()
            words = text.split(' ')
            words = [morph.parse(word)[0].normal_form for word in words]

            return words

        def get_01_xsamples(tok_agreem, tags_index):
            tags01 = []
            indexes = []

            for agreement in tok_agreem:
                tag_place = [0, 0, 0, 0, 0, 0]
                for ex in agreement:
                    if ex in tags_index:
                        place = np.argwhere(tags_index == ex)
                        if len(place) != 0:
                            if place[0][0] < 6:
                                tag_place[place[0][0]] = 1
                            else:
                                tag_place[place[0][0] - 6] = 0
                    else:
                        tags01.append(tag_place.copy())
                        indexes.append(ex)

            return indexes, tags01

        def reverse_index(clean_voc, x):

            reverse_word_map = dict(map(reversed, clean_voc.items()))
            words = [reverse_word_map.get(letter) for letter in x]

            return words

        def get_set_from_indexes(word_indexes, x_len, step):
            x_batch = []
            words_len = len(word_indexes)
            index = 0

            while index + x_len <= words_len:
                x_batch.append(word_indexes[index:index + x_len])
                index += step

            return x_batch

        def get_sets(model, sen_i, tag_i):
            x_vector = []
            idx = 0
            progress_bar = tqdm(sen_i, ncols=800)
            progress_bar.set_description('Формирование массивов')
            for text in progress_bar:
                tmp = []
                for word in text:
                    tmp.append(model[word])
                x_vector.append(tmp)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total and idx == len(sen_i):
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)

            return np.array(x_vector), np.array(tag_i)

        cur_time = time()
        if folder_name == None:
            folder_name = self.file_folder
        self.source_datatype = self._set_datatype(text=True)
        self.num_classes = num_classes
        text_list = []
        folder_list = sorted(os.listdir(folder_name))
        for folder in folder_list:
            file_list = [i for i in sorted(os.listdir(os.path.join(folder_name, folder)))]
            progress_bar = tqdm(file_list, ncols=800)
            progress_bar.set_description(f'Загрузка файлов из папки {folder}')
            idx = 0
            for file in progress_bar:
                txt = read_text(os.path.join(folder_name, folder, file))
                if txt != '':
                    text_list.append(txt)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    self.Exch.print_progress_bar(progress_bar_status)

        words = []
        idx = 0
        progress_bar = tqdm(text_list, ncols=800)
        progress_bar.set_description(f'Составление общего списка слов')
        for txt in progress_bar:
            words.append(text_to_words(txt))
            if self.django_flag:
                idx += 1
                progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                       f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                self.Exch.print_progress_bar(progress_bar_status)

        tokenizer = Tokenizer(num_words=max_words_count, filters='–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff',
                              lower=True, split=' ', char_level=False, oov_token='unknown')
        tokenizer.fit_on_texts(words)

        clean_voc = {}
        for item in tokenizer.word_index.items():
            clean_voc[item[0]] = item[1]
        tok_agreem = tokenizer.texts_to_sequences(words)

        tags_index = ['<s' + str(i) + '>' for i in range(1, num_classes + 1)]
        closetags = ['</s' + str(i) + '>' for i in range(1, num_classes + 1)]
        tags_index.extend(closetags)

        tags_index = np.array([clean_voc[i] for i in tags_index])
        x_data, y_data = get_01_xsamples(tok_agreem, tags_index)
        decoded_text = reverse_index(clean_voc, x_data)
        x_Train = get_set_from_indexes(decoded_text, x_len, step)
        y_Train = get_set_from_indexes(y_data, x_len, step)
        if not self.django_flag:
            print('Формирование word2vec')
        model_gensim = word2vec.Word2Vec(x_Train, size=embedding_size, window=10, min_count=1, workers=10, iter=10)
        X, Y = get_sets(model_gensim, x_Train, y_Train)
        self.source_shape = X.shape[1:]
        self.input_shape = X.shape[1:]
        self.input_datatype = self._set_datatype(shape=X.shape)

        self.x_Train, self.x_Val, self.y_Train, self.y_Val = train_test_split(X, Y, test_size=self.divide_ratio[0][1],
                                                                              shuffle=True)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size,
                                                                                shuffle=True)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def voice_recognition(self, sample_rate, length, folder_name=None, **options):

        """
        Prepare data for processing in neural network for voice analysis.

        Examples:
            voice_rec = trds.DTS()
            voice_rec.voice_recognition(sample_rate=22050, test_size=0.2, length=11025)

        Args:
            sample_rate (int): time series sampling rate (usually = 22050);
            length (int): size of section on which data will be splitted;
            folder_name (str): path to folder with commands.

        **options:
            net = 'conv'; adds dimension for processing on convolution layers
            one_hot_encoding = True; transform to OHE vector

        Returns:
            Arrays prepared for processing in neural network.
        """

        def get_labels(path):

            labels = sorted(os.listdir(path))
            label_indices = np.arange(0, len(labels))

            return labels, label_indices, utils.to_categorical(label_indices)

        def wav2mfcc(file_path, length=11025, step=2205):
            out_mfcc = []
            out_audio = []
            y, sr = librosa.load(file_path)

            while (len(y) >= length):
                section = y[:length]
                section = np.array(section)
                out_mfcc.append(librosa.feature.mfcc(section, sr))
                out_audio.append(section)
                y = y[step:]

            out_mfcc = np.array(out_mfcc)
            out_audio = np.array(out_audio)

            return out_mfcc, out_audio

        cur_time = time()
        if folder_name == None:
            folder_name = self.file_folder
        feature_dim_1 = 20
        feature_dim_2 = int(.5 * sample_rate)
        step_mfcc = int(.02 * sample_rate)
        channel = 1
        self.classes_names = [folder for folder in sorted(os.listdir(folder_name))]
        self.num_classes = len(self.classes_names)
        labels, indices, _ = get_labels(folder_name)

        Y = np.array([])
        for i, label in enumerate(labels):
            mfcc_vectors = []
            wavfiles = [os.path.join(folder_name, label, wavfile) for wavfile in
                        sorted(os.listdir(os.path.join(folder_name, label)))]
            progress_bar = tqdm(wavfiles, ncols=800)
            progress_bar.set_description(f'Загрузка из папки: {os.path.join(folder_name, label)}')
            idx = 0
            for wavfile in progress_bar:
                mfcc, _ = wav2mfcc(wavfile, length=length, step=step_mfcc)
                if (mfcc.shape[0] != 0):
                    mfcc_vectors.extend(mfcc)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total and i + 1 == len(labels):
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)
            mfcc_vectors = np.array(mfcc_vectors)
            try:
                X = np.vstack((X, mfcc_vectors))
            except NameError:
                X = mfcc_vectors
            Y = np.append(Y, np.full(mfcc_vectors.shape[0], fill_value=(i)))

        self.source_shape = X.shape[1:]
        self.source_datatype = self._set_datatype(shape=X.shape)

        if 'net' in options.keys() and options['net'] == 'conv':
            X = X[..., None]

        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] == True:
            Y = utils.to_categorical(Y)

        self.input_shape = X.shape[1:]
        self.input_datatype = self._set_datatype(shape=X.shape)

        self.x_Train, self.x_Val, self.y_Train, self.y_Val = train_test_split(X, Y, test_size=self.divide_ratio[0][1],
                                                                              shuffle=True)

        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size,
                                                                                shuffle=True)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def data_regression(self, filename, x_len, val_len, **options):
        """
        Prepare data for processing in neural network for regression analysis.

        Examples:
            regression = trds.DTS()
            regression.data_regression(filepath=path_to_file, x_len=20, val_len=1000, timeseriesgenerator=True)

        Args:
            filename (str): name of file;
            x_len (int): number of periods for frame;
            val_len (int): size of validation data

        **options:
            x_cols:  columns for usage as x_train/x_test;
            y_col:  column for usage as y_train/y_test;
            graph = True:  build a graph;
            timeseriesgenerator = True; apply TimeseriesGenerator;
            timeseries_batch_size:  size of  batch_size for TimeseriesGenerator;
            scaler = 'MinMaxScaler': apply MinMaxScaler;
            scaler = 'StandardScaler': apply StandardScaler.

        Returns:
            Arrays or TimeSeriesGenerator for processing in neural network.
        """

        folder_name = self.file_folder
        if self.name == 'sber':
            data = pd.read_csv(os.path.join(folder_name, filename), sep='\t', header=None)
        else:
            data = pd.read_csv(os.path.join(folder_name, filename))

        if not self.django_flag:
            display(data.head())

        if 'x_cols' not in options.keys() and 'y_col' not in options.keys():
            x_cols = []
            y_col = []
            x_channel = input('Введите через пробел названия колонок для x_train: ')
            y_channel = input('Введите название одной колонки для y_train: ')
            for col_name in x_channel.split(' '):
                x_cols.append(col_name)
            for col_name in y_channel.split(' '):
                y_col.append(col_name)
        else:
            x_cols = options['x_cols']
            y_col = options['y_col']
        if len(y_col) > 1:
            print_error_msg = 'Нельзя выбрать две колонки для Y для задачи регрессии. Выберите одну.'
            if self.django_flag:
                self.Exch.print_error(('Error', print_error_msg))
            else:
                assert not len(y_col) > 1, print_error_msg
        self.classes_names = y_col
        self.num_classes = len(y_col)
        cur_time = time()
        x_data = data[x_cols]
        y_data = data[y_col]
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if not self.django_flag:
            print(f'Количество примеров: {data.shape[0]}')
            if 'graph' in options.keys() and options['graph'] == True:
                plt.figure(figsize=(20, 4))
                plt.grid(True, alpha=0.5)
                for i in range(len(x_cols)):
                    plt.plot(x_data[:, i], label=[x_cols + y_col][0][i])
                plt.plot(y_data[:, 0], label=y_col[0])
                plt.ylabel('Цена')
                plt.legend()
                plt.show()

        train_len = data.shape[0] - val_len

        self.x_Train, self.x_Val = x_data[:train_len, :], x_data[train_len + x_len + 2:, :]
        self.y_Train, self.y_Val = np.reshape(y_data[:train_len, 0], (-1, 1)), np.reshape(
            y_data[train_len + x_len + 2:, 0],
            (-1, 1))
        self.source_shape = self.x_Train.shape[1:]
        self.source_datatype = self._set_datatype(shape=self.x_Train.shape)
        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.y_Val, self.y_Test = train_test_split(self.x_Val, self.y_Val,
                                                                                test_size=1 - split_size, shuffle=True)

        arrays = ['x_Train', 'x_Val', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
        if 'scaler' in options.keys():
            if options['scaler'].lower() == 'MinMaxScaler'.lower():
                x_scaler = MinMaxScaler()
                y_scaler = MinMaxScaler()
            elif options['scaler'].lower() == 'StandardScaler'.lower():
                x_scaler = StandardScaler()
                y_scaler = StandardScaler()
            x_scaler.fit(self.x_Train)
            y_scaler.fit(self.y_Train)
            for item in arrays:
                if hasattr(self, item):
                    if item in arrays[:3]:
                        self.__dict__[item] = x_scaler.transform(self.__dict__[item])
                    elif item in arrays[3:]:
                        self.__dict__[item] = y_scaler.transform(self.__dict__[item])
            self.x_Scaler = x_scaler
            self.y_Scaler = y_scaler
        self.input_shape = self.x_Train.shape[1:]
        self.input_datatype = self._set_datatype(shape=self.x_Train.shape)
        if 'timeseriesgenerator' in options.keys() and options['timeseriesgenerator'] == True:
            for item in arrays[:3]:
                if hasattr(self, item):
                    self.__dict__[f'{item[2:]}_data_gen'] = TimeseriesGenerator(self.__dict__[item], self.__dict__[
                        arrays[arrays.index(item) + 3]], length=x_len, stride=1, batch_size=options[
                        'timeseries_batch_size'])
            if not self.django_flag:
                print(f'Формирование генератора завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
        else:
            if not self.django_flag:
                print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
                for item in arrays:
                    if hasattr(self, item):
                        print(f'Размерность {item}: {self.__dict__[item].shape}')
        self.dts_prepared = True

        return self

    def custom_dataset(self, *Data, **options):

        self.X = Data[0]
        self.Y = Data[1]

        self.source = 'custom_dataset'
        for key in list(self.X.keys()):
            self.source_shape.append(self.X[key][0].shape[1:])
            self.source_datatype.append(self._set_datatype(shape=self.X[key][0].shape))

        for key in list(self.X.keys()):
            self.input_shape.append(self.X[key][0].shape[1:])
            self.input_datatype.append(self._set_datatype(shape=self.X[key][0].shape))

        self.dts_prepared = True

        return self

    def flat_parser(self, **options):

        def getRoomsCount(d, maxRoomCount):
            roomsCountStr = d[0]

            roomsCount = 0
            try:
                roomsCount = int(roomsCountStr)
                if (roomsCount > maxRoomCount):
                    roomsCount = maxRoomCount
            except:
                if (roomsCountStr == roomsCountStr):
                    if ("Ст" in roomsCountStr):
                        roomsCount = maxRoomCount + 1

            return roomsCount

        def getRoomsCountCategory(d, maxRoomCount):
            roomsCount = getRoomsCount(d, maxRoomCount)
            roomsCount = utils.to_categorical(roomsCount, maxRoomCount + 2)
            return roomsCount

        def getRoomsCount(d, maxRoomCount):
            roomsCountStr = d[0]

            roomsCount = 0
            try:
                roomsCount = int(roomsCountStr)
                if (roomsCount > maxRoomCount):
                    roomsCount = maxRoomCount
            except:
                if (roomsCountStr == roomsCountStr):
                    if ("Ст" in roomsCountStr):
                        roomsCount = maxRoomCount + 1

            return roomsCount

        def getRoomsCountCategory(d, maxRoomCount):
            roomsCount = getRoomsCount(d, maxRoomCount)
            roomsCount = utils.to_categorical(roomsCount, maxRoomCount + 2)
            return roomsCount

        def getMetro(d, allMetroNames):
            metroStr = d[1]
            metro = 0

            if (metroStr in allMetroNames):
                metro = allMetroNames.index(metroStr) + 1

            return metro

        def getMetroType(d):
            metroTypeStr = d[1]
            metroTypeClasses = 5
            metroType = metroTypeClasses - 1

            metroNamesInsideCircle = ["Площадь Революции", "Арбатская", "Смоленская", "Красные Ворота", "Чистые пруды",
                                      "Лубянка", "Охотный Ряд", "Библиотека имени Ленина", "Кропоткинская",
                                      "Сухаревская",
                                      "Тургеневская", "Китай-город", "Третьяковская", "Трубная", "Сретенский бульвар",
                                      "Цветной бульвар", "Чеховская", "Боровицкая", "Полянка", "Маяковская", "Тверская",
                                      "Театральная", "Новокузнецкая", "Пушкинская", "Кузнецкий Мост", "Китай-город",
                                      "Александровский сад"]

            metroNamesCircle = ["Киевская", "Парк Культуры", "Октябрьская", "Добрынинская", "Павелецкая", "Таганская",
                                "Курская", "Комсомольская", "Проспект Мира", "Новослободская", "Белорусская",
                                "Краснопресненская"]

            metroNames13FromCircle = ["Бауманская", "Электрозаводская", "Семёновская", "Площадь Ильича", "Авиамоторная",
                                      "Шоссе Энтузиастов", "Римская", "Крестьянская Застава", "Дубровка",
                                      "Пролетарская",
                                      "Волгоградский проспект", "Текстильщики", "Автозаводская", "Технопарк",
                                      "Коломенская",
                                      "Тульская", "Нагатинская", "Нагорная", "Шаболовская", "Ленинский проспект",
                                      "Академическая", "Фрунзенская", "Спортивная", "Воробьёвы горы", "Студенческая",
                                      "Кутузовская", "Фили", "Парк Победы", "Выставочная", "Международная",
                                      "Улица 1905 года",
                                      "Беговая", "Полежаевская", "Динамо", "Аэропорт", "Сокол", "Деловой центр",
                                      "Шелепиха",
                                      "Хорошёвская", "ЦСКА", "Петровский парк", "Савёловская", "Дмитровская",
                                      "Тимирязевская",
                                      "Достоевская", "Марьина Роща", "Бутырская", "Фонвизинская", "Рижская",
                                      "Алексеевская",
                                      "ВДНХ", "Красносельская", "Сокольники", "Преображенская площадь"]

            metroNames48FromCircle = ["Партизанская", "Измайловская", "Первомайская", "Щёлковская", "Новокосино",
                                      "Новогиреево",
                                      "Перово", "Кузьминки", "Рязанский проспект", "Выхино", "Лермонтовский проспект",
                                      "Жулебино", "Партизанская", "Измайловская", "Первомайская", "Щёлковская",
                                      "Новокосино",
                                      "Новогиреево", "Перово", "Кузьминки", "Рязанский проспект", "Выхино",
                                      "Лермонтовский проспект", "Жулебино", "Улица Дмитриевского", "Кожуховская",
                                      "Печатники",
                                      "Волжская", "Люблино", "Братиславская", "Коломенская", "Каширская",
                                      "Кантемировская",
                                      "Царицыно", "Орехово", "Севастопольская", "Чертановская", "Южная", "Пражская",
                                      "Варшавская", "Профсоюзная", "Новые Черёмушки", "Калужская", "Беляево",
                                      "Коньково",
                                      "Университет", "Багратионовская", "Филёвский парк", "Пионерская", "Кунцевская",
                                      "Молодёжная", "Октябрьское Поле", "Щукинская", "Спартак", "Тушинская",
                                      "Сходненская",
                                      "Войковская", "Водный стадион", "Речной вокзал", "Беломорская", "Ховрино",
                                      "Петровско-Разумовская", "Владыкино", "Отрадное", "Бибирево", "Алтуфьево",
                                      "Фонвизинская",
                                      "Окружная", "Верхние Лихоборы", "Селигерская", "ВДНХ", "Ботанический сад",
                                      "Свиблово",
                                      "Бабушкинская", "Медведково", "Преображенская площадь", "Черкизовская",
                                      "Бульвар Рокоссовского"]

            if (metroTypeStr in metroNamesInsideCircle):
                metroType = 0
            if (metroTypeStr in metroNamesCircle):
                metroType = 1
            if (metroTypeStr in metroNames13FromCircle):
                metroType = 2
            if (metroTypeStr in metroNames48FromCircle):
                metroType = 3

            metroType = utils.to_categorical(metroType, metroTypeClasses)
            return metroType

        def getMetroDistance(d):
            metroDistanceStr = d[2]
            metroDistance = 0
            metroDistanceType = 0

            if (metroDistanceStr == metroDistanceStr):
                if (len(metroDistanceStr) > 0):
                    if (metroDistanceStr[-1] == "п"):
                        metroDistanceType = 1
                    elif (metroDistanceStr[-1] == "т"):
                        metroDistanceType = 2

                    metroDistanceStr = metroDistanceStr[:-1]
                    try:
                        metroDistance = int(metroDistanceStr)
                        if (metroDistance < 3):
                            metroDistance = 1
                        elif (metroDistance < 6):
                            metroDistance = 2
                        elif (metroDistance < 10):
                            metroDistance = 3
                        elif (metroDistance < 15):
                            metroDistance = 4
                        elif (metroDistance < 20):
                            metroDistance = 5
                        else:
                            metroDistance = 6
                    except:
                        metroDistance = 0

            metroDistanceClasses = 7

            if (metroDistanceType == 2):
                metroDistance += metroDistanceClasses
            if (metroDistanceType == 0):
                metroDistance += 2 * metroDistanceClasses

            metroDistance = utils.to_categorical(metroDistance, 3 * metroDistanceClasses)
            return metroDistance

        def getHouseTypeAndFloor(d):
            try:
                houseStr = d[3]
            except:
                houseStr = ""

            houseType = 0
            floor = 0
            floors = 0
            isLastFloor = 0

            if (houseStr == houseStr):
                if (len(houseStr) > 1):

                    try:
                        slashIndex = houseStr.index("/")
                    except:
                        print(houseStr)

                    try:
                        spaceIndex = houseStr.index(" ")
                    except:
                        print(houseStr)

                    floorStr = houseStr[:slashIndex]
                    floorsStr = houseStr[slashIndex + 1:spaceIndex]
                    houseTypeStr = houseStr[spaceIndex + 1:]

                    try:
                        floor = int(floorStr)
                        floorSave = floor
                        if (floorSave < 5):
                            floor = 2
                        if (floorSave < 10):
                            floor = 3
                        if (floorSave < 20):
                            floor = 4
                        if (floorSave >= 20):
                            floor = 5
                        if (floorSave == 1):
                            floor = 1

                        if (floor == floors):
                            isLastFloor = 1
                    except:
                        floor = 0

                    try:
                        floors = int(floorsStr)
                        floorsSave = floors
                        if (floorsSave < 5):
                            floors = 1
                        if (floorsSave < 10):
                            floors = 2
                        if (floorsSave < 20):
                            floors = 3
                        if (floorsSave >= 20):
                            floors = 4
                    except:
                        floors = 0

                    if (len(houseTypeStr) > 0):
                        if ("М" in houseTypeStr):
                            houseType = 1
                        if ("К" in houseTypeStr):
                            houseType = 2
                        if ("П" in houseTypeStr):
                            houseType = 3
                        if ("Б" in houseTypeStr):
                            houseType = 4
                        if ("?" in houseTypeStr):
                            houseType = 5
                        if ("-" in houseTypeStr):
                            houseType = 6

                floor = utils.to_categorical(floor, 6)
                floors = utils.to_categorical(floors, 5)
                houseType = utils.to_categorical(houseType, 7)

            return floor, floors, isLastFloor, houseType

        def getBalcony(d):
            balconyStr = d[4]
            balconyVariants = ['Л', 'Б', '2Б', '-', '2Б2Л', 'БЛ', '3Б', '2Л', 'Эрк', 'Б2Л', 'ЭркЛ', '3Л', '4Л', '*Л',
                               '*Б']
            if (balconyStr == balconyStr):
                balcony = balconyVariants.index(balconyStr) + 1
            else:
                balcony = 0
            balcony = utils.to_categorical(balcony, 16)

            return balcony

        def getWC(d):
            wcStr = d[5]
            wcVariants = ['2', 'Р', 'С', '-', '2С', '+', '4Р', '2Р', '3С', '4С', '4', '3', '3Р']
            if (wcStr == wcStr):
                wc = wcVariants.index(wcStr) + 1
            else:
                wc = 0
            wc = utils.to_categorical(wc, 14)

            return wc

        def getArea(d):
            areaStr = d[6]

            if ("/" in areaStr):
                slashIndex = areaStr.index("/")
                try:
                    area = float(areaStr[:slashIndex])
                except:
                    area = 0
            else:
                area = 0

            return area

        def getCost(d):
            costStr = d[7]
            try:
                cost = float(costStr)
            except:
                cost = 0

            return cost

        def getComment(d):
            commentStr = d[-1]

            return commentStr

        def getAllParameters(d, allMetroNames):
            roomsCountType = getRoomsCountCategory(d, 30)
            metro = getMetro(d, allMetroNames)
            metroType = getMetroType(d)
            metroDistance = getMetroDistance(d)
            floor, floors, isLastFloor, houseType = getHouseTypeAndFloor(d)
            balcony = getBalcony(d)
            wc = getWC(d)
            area = getArea(d)

            out = list(roomsCountType)
            out.append(metro)
            out.extend(metroType)
            out.extend(metroDistance)
            out.extend(floor)
            out.extend(floors)
            out.append(isLastFloor)
            out.extend(houseType)
            out.extend(balcony)
            out.extend(wc)
            out.append(area)

            return out

        def getXTrain(data):
            allMertroNames = list(df["Метро / ЖД станции"].unique())
            idx = 0
            xTrain = []
            progress_bar = tqdm(data, ncols=800)
            progress_bar.set_description(f'Формирование X')
            for d in progress_bar:
                xTrain.append(getAllParameters(d, allMertroNames))
                if self.django_flag:
                    idx += 1
                    if idx % 1202 == 0:
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)

            xTrain = np.array(xTrain)

            return xTrain

        def getYTrain(data):

            idx = 0
            yTrain = []
            progress_bar = tqdm(data, ncols=800)
            progress_bar.set_description(f'Формирование Y')
            for d in progress_bar:
                yTrain.append(getCost(d))
                if self.django_flag:
                    idx += 1
                    if idx % 1202 == 0:
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total and idx == len(data):
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)
            yTrain = np.array(yTrain)

            return yTrain

        def text2Words(text):
            text = text.replace(".", "")
            text = text.replace("—", "")
            text = text.replace(",", "")
            text = text.replace("!", "")
            text = text.replace("?", "")
            text = text.replace("…", "")
            text = text.lower()

            words = []
            currWord = ""

            for symbol in text:

                if (symbol != "\ufeff"):
                    if (symbol != " "):
                        currWord += symbol
                    else:
                        if (currWord != ""):
                            words.append(currWord)
                            currWord = ""

            if (currWord != ""):
                words.append(currWord)

            return words

        def createVocabulary(allWords):
            wCount = dict.fromkeys(allWords, 0)

            for word in allWords:
                wCount[word] += 1

            wordsList = list(wCount.items())
            wordsList.sort(key=lambda i: i[1], reverse=1)

            sortedWords = []

            for word in wordsList:
                sortedWords.append(word[0])

            wordIndexes = dict.fromkeys(allWords, 0)
            for word in wordIndexes.keys():
                wordIndexes[word] = sortedWords.index(word) + 1

            return wordIndexes

        def words2Indexes(words, vocabulary, maxWordsCount):
            wordsIndexes = []

            for word in words:

                wordIndex = 0
                wordInVocabulary = word in vocabulary

                if (wordInVocabulary):
                    index = vocabulary[word]
                    if (index < maxWordsCount):
                        wordIndex = index

                wordsIndexes.append(wordIndex)

            return wordsIndexes

        def changeXTo01(trainVector, wordsCount):
            out = np.zeros(wordsCount)
            for x in trainVector:
                out[x] = 1

            return out

        def changeSetTo01(trainSet, wordsCount):
            out = []

            for x in trainSet:
                out.append(
                    changeXTo01(x, wordsCount))

            return np.array(out)

        def getXTrainComments(data):
            xTrainComments = []
            allTextComments = ""

            for d in data:
                currText = getComment(d)
                try:
                    if currText == currText:
                        allTextComments += currText + " "
                except:
                    currText = "Нет комментария"
                xTrainComments.append(currText)

            xTrainComments = np.array(xTrainComments)

            return (xTrainComments, allTextComments)

        def changeSetToIndexes(xTrainComments, vocabulary, maxWordsCount):

            xTrainCommentsIndexes = []

            idx = 0
            progress_bar = tqdm(xTrainComments, ncols=800)
            progress_bar.set_description(f'Формирование X-Comments')
            for text in progress_bar:
                currWords = text2Words(text)
                currIndexes = words2Indexes(currWords, vocabulary, maxWordsCount)
                currIndexes = np.array(currIndexes)
                xTrainCommentsIndexes.append(currIndexes)
                if self.django_flag:
                    idx += 1
                    if idx % 1202 == 0:
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        self.Exch.print_progress_bar(progress_bar_status)

            xTrainCommentsIndexes = np.array(xTrainCommentsIndexes)
            xTrainCommentsIndexes = changeSetTo01(xTrainCommentsIndexes, maxWordsCount)
            return xTrainCommentsIndexes

        def changeSetToIndexesCrop(xTrainComments, vocabulary, maxWordsCount, cropLen):
            xTrainCommentsIndexes = []

            for text in xTrainComments:
                currWords = text2Words(text)
                currIndexes = words2Indexes(currWords, vocabulary, maxWordsCount)
                currIndexes = np.array(currIndexes)
                xTrainCommentsIndexes.append(currIndexes)

            xTrainCommentsIndexes = np.array(xTrainCommentsIndexes)
            xTrainCommentsIndexes = pad_sequences(xTrainCommentsIndexes,
                                                  maxlen=cropLen)
            return xTrainCommentsIndexes

        cur_time = time()
        data = pd.read_csv(f'{self.file_folder}/moscow.csv', sep=';')
        df = data.copy()
        data = data.values
        oneRoomMask = [getRoomsCount(d, 30) == 1 for d in data]
        data1 = data[oneRoomMask]

        xTrain = getXTrain(data1)

        xTrainC, allTextComments = getXTrainComments(data1)
        allWords = text2Words(allTextComments)
        allWords = allWords[::10]
        vocabulary = createVocabulary(allWords)
        xTrainC01 = changeSetToIndexes(xTrainC, vocabulary, 2000)

        yTrain = getYTrain(data1)

        self.x_Scaler = StandardScaler()
        self.x_Scaler.fit(xTrain[:, -1].reshape(-1, 1))
        xTrainScaled = xTrain.copy()
        xTrainScaled[:, -1] = self.x_Scaler.transform(xTrain[:, -1].reshape(-1, 1)).flatten()

        self.y_Scaler = StandardScaler()
        self.y_Scaler.fit(yTrain.reshape(-1, 1))
        yTrainScaled = self.y_Scaler.transform(yTrain.reshape(-1, 1))

        self.x_Train, self.x_Val, self.x_TrainC01, self.x_ValC01, self.y_Train, self.y_Val = train_test_split(
            xTrainScaled, xTrainC01,
            yTrainScaled,
            test_size=self.divide_ratio[0][1], shuffle=True)
        self.source_shape = self.x_Train.shape[1:]
        self.source_datatype = self._set_datatype(shape=self.x_Train.shape)
        if 'test' in options.keys() and options['test'] == True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            self.x_Val, self.x_Test, self.x_ValC01, self.x_TestC01, self.y_Val, self.y_Test = train_test_split(
                self.x_Val, self.x_ValC01, self.y_Val,
                test_size=1 - split_size,
                shuffle=True)
        self.input_shape = self.x_Train.shape[1:]
        self.input_datatype = self._set_datatype(shape=self.x_Train.shape)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            arrays = ['x_Train', 'x_Val', 'x_TrainC01', 'x_ValC01', 'x_Test', 'y_Train', 'y_Val', 'y_Test']
            for item in arrays:
                if hasattr(self, item):
                    print(f'Размерность {item}: {self.__dict__[item].shape}')

        return self

    def as_array(self):

        """
        Returns arrays from class.

        Example:
              (x_Train, y_Train), (x_Val, y_Val) = trds.DTS().as_array()
              or
              (x_Train, y_Train), (x_Val, y_Val), (x_Test, y_Test) = trds.DTS().as_array()

        Returns:
            Arrays prepared for processing in neural network.

        """
        if hasattr(self, 'x_Test') and hasattr(self, 'y_Test'):
            return (self.x_Train, self.y_Train), (self.x_Val, self.y_Val), (self.x_Test, self.y_Test)
        else:
            return (self.x_Train, self.y_Train), (self.x_Val, self.y_Val)

    def get_datasets_dict(self):

        datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters', 'sber',
                    'автомобили', 'автомобили_3', 'самолеты', 'умный_дом', 'договоры', 'трейдинг',
                    'квартиры', 'болезни', 'заболевания', 'губы', 'жанры_музыки']
        datasets_dict = {}
        for data in datasets:
            datasets_dict[data] = [self._set_tag(data), self._set_language(data), self._set_source(data)]

        return datasets_dict

    def get_datasets_methods(self):

        func_list = [method for method in dir(self) if
                     callable(getattr(self, method)) and not method.startswith("_") and not method.startswith("__")]
        datasets_methods = {}
        for func in func_list:
            for tupl in getmembers(self):
                if tupl[0] == func:
                    params = str(signature(tupl[1]))
                    idx1 = params.index('(')
                    idx2 = params.index(')')
                    params = params[idx1 + 1:idx2].split(', ')
                    datasets_methods[func] = params

        return datasets_methods

    def inverse_data(self, array):

        inversed_data = []

        return inversed_data

    def prepare_dataset(self, **options):

        if not 'dataset_name' in options.keys() or not 'task_type' in options.keys():
            print_error_msg = 'Отсутствует один из необходимых параметров dataset_name="" или task_type=""'
            self.Exch.print_error(('Error', print_error_msg))
            if not self.django_flag:
                assert 'dataset_name' in options.keys() or 'task_type' in options.keys(), print_error_msg

        if options['dataset_name'] == 'boston_housing':
            if options['task_type'] == 'regression':
                self.keras_datasets(options['dataset_name'], scaler='StandardScaler', test=True)
            else:
                if not self.django_flag:
                    print('Для датасета', options['dataset_name'], 'доступен только следующий тип задачи:',
                          self._set_tag(options['dataset_name'])[1:])
        elif options['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
            if options['task_type'] == 'classification':
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, scaler='MinMaxScaler', net='conv', test=True)
            else:
                if not self.django_flag:
                    print('Для датасета', options['dataset_name'], 'доступен только следующий тип задачи:',
                          self._set_tag(options['dataset_name'])[1:])
        elif options['dataset_name'] == 'imdb':
            if options['task_type'] == 'classification':
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, test=True)
            else:
                if not self.django_flag:
                    print('Для датасета', options['dataset_name'], 'доступен только следующий тип задачи:',
                          self._set_tag(options['dataset_name'])[1:])
        elif options['dataset_name'] == 'reuters':
            if options['task_type'] == 'classification':
                self.keras_datasets(options['dataset_name'], test=True)
            else:
                if not self.django_flag:
                    print('Для датасета', options['dataset_name'], 'доступен только следующий тип задачи:',
                          self._set_tag(options['dataset_name'])[1:])

        else:
            self.load_data(options['dataset_name'])
            if options['task_type'] == 'classification':
                if 'images' in self.tags:
                    self.image_classification((54, 96), one_hot_encoding=True, scaler='MinMaxScaler')
                elif 'text' in self.tags:
                    if 'заболевания' in self.name:
                        max_words_count = 20000
                        x_len = 100
                        step = 30
                    self.text_classification(max_words_count, x_len, step, one_hot_encoding=True)
            elif options['task_type'] == 'segmentation':
                if 'images' in self.tags:
                    if 'самолеты' in self.name:
                        classes = {'небо': [0, 0, 0], 'самолёт': [255, 0, 0]}
                        range = 50
                    elif 'губы' in self.name:
                        classes = {'фон': [0, 0, 0], 'губы': [0, 255, 0]}
                        range = 10
                    self.image_segmentation((44, 60), classes, range, scaler='MinMaxScaler')
                elif 'text' in self.tags:
                    self.text_segmentation(max_words_count=20000, x_len=256, step=30, embedding_size=300, num_classes=6)
            elif options['task_type'] == 'recognition':
                if 'audio' in self.tags:
                    if 'умный_дом' in self.name:
                        self.file_folder = self.file_folder + '/comands'
                        s_rate = 22050
                        len = 11025
                    self.voice_recognition(s_rate, len, net='conv', one_hot_encoding=True)
            elif options['task_type'] == 'regression':
                if 'трейдинг' in self.name:
                    # self.data_regression('shares/GAZP_1d_from_MOEX.txt', x_len=80, val_len=300, graph=True,
                    #                      timeseriesgenerator=True, test=True, timeseries_batch_size=50,
                    #                      x_cols=['<OPEN>', '<HIGH>', '<LOW>'], y_col=['<CLOSE>'])
                    self.data_regression('shares/GAZP_1d_from_MOEX.txt', x_len=80, val_len=300, graph=True,
                                         test=True, x_cols=['<OPEN>', '<HIGH>', '<LOW>'], y_col=['<CLOSE>'])
                if 'квартиры' in self.name:
                    self.flat_parser()
        return self


if __name__ == '__main__':
    # obj1 = DTS().image_segmentation()
    pass
