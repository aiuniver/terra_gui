from terra_ai.data.datasets.dataset import DatasetPathsData
# from terra_ai.data.datasets.extra import LayerScalerImageChoice

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from gensim.models.word2vec import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer


class TerraImageScaler:

    def __init__(self, shape=(176, 240), min_max=(0, 1)):

        self.shape: tuple = shape
        self.trained_values: dict = {'red': {'min': np.full(shape, 255, dtype='uint8'),
                                             'max': np.zeros(shape, dtype='uint8')},
                                     'green': {'min': np.full(shape, 255, dtype='uint8'),
                                               'max': np.zeros(shape, dtype='uint8')},
                                     'blue': {'min': np.full(shape, 255, dtype='uint8'),
                                              'max': np.zeros(shape, dtype='uint8')}}
        self.range = min_max

        pass

    def fit(self, img):
        for i, channel in enumerate(['red', 'green', 'blue']):
            min_mask = img[:, :, i] < self.trained_values[channel]['min']
            max_mask = img[:, :, i] > self.trained_values[channel]['max']
            self.trained_values[channel]['min'][min_mask] = img[:, :, i][min_mask]
            self.trained_values[channel]['max'][max_mask] = img[:, :, i][max_mask]

    def transform(self, img):

        channels = ['red', 'green', 'blue']
        transformed_img = []
        for ch in channels:
            x = img[:, :, channels.index(ch)]
            y1 = np.full(self.shape, self.range[0])
            y2 = np.full(self.shape, self.range[1])
            x1 = self.trained_values[ch]['min']
            x2 = self.trained_values[ch]['max']
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
            transformed_img.append(y)

        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < self.range[0]] = self.range[0]
        array[array > self.range[1]] = self.range[1]

        return array

    def inverse_transform(self, img):

        channels = ['red', 'green', 'blue']
        transformed_img = []
        for ch in channels:
            x = img[:, :, channels.index(ch)]
            x1 = np.full(self.shape, self.range[0])
            x2 = np.full(self.shape, self.range[1])
            y1 = self.trained_values[ch]['min']
            y2 = self.trained_values[ch]['max']
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
            transformed_img.append(y)

        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < 0] = 0
        array[array > 255] = 255

        return array.astype('uint8')


class CreatePreprocessing(object):

    def __init__(self, dataset_path=None):

        if dataset_path:
            self.paths = DatasetPathsData(basepath=dataset_path)
        self.preprocessing = {}

    def load_preprocesses(self, keys):
        for key in keys:
            preprocess = {}
            for param in ['augmentation', 'scaler', 'tokenizer', 'word2vec']:
                if self.paths.__dict__[param]:
                    if os.path.isfile(os.path.join(self.paths.__dict__[param], f'{key}.gz')):
                        preprocess[f'object_{param}'] = joblib.load(os.path.join(self.paths.__dict__[param], f'{key}.gz'))
                    else:
                        preprocess[f'object_{param}'] = None
            self.preprocessing[key] = preprocess
        pass

    def create_dull(self, put_id: int):

        self.preprocessing[put_id] = {'dull': None}

    def create_scaler(self, put_id: int, array=None, **options):
        scaler = None
        if "MinMaxScaler_cols" in options.keys() or 'trend' in options.keys():
            array = pd.DataFrame(array)

        if options['scaler'] != 'no_scaler':
            if options['scaler'] == 'min_max_scaler':
                scaler = MinMaxScaler()
                array = np.array(array).reshape(-1, 1) if isinstance(array, np.ndarray) or \
                                                            isinstance(array, pd.DataFrame) else np.array([[0], [255]])
                scaler.fit(array)
            elif options['scaler'] == 'standard_scaler':
                scaler = StandardScaler()
                array = np.array(array).reshape(-1, 1)
                scaler.fit(array)
            elif options['scaler'] == 'terra_image_scaler':
                scaler = TerraImageScaler(shape=(176, 220))  # УКАЗАТЬ РАЗМЕРНОСТЬ
                for elem in array:
                    scaler.fit(elem)
            self.preprocessing[put_id] = {'object_scaler': scaler}

        elif ("MinMaxScaler_cols" in options.keys() and options["MinMaxScaler_cols"]) or \
                ("StandardScaler_cols" in options.keys() and options["StandardScaler_cols"]):
            self.preprocessing[put_id] = {"object_scaler": {}}
            if options["MinMaxScaler_cols"]:
                for i in options["MinMaxScaler_cols"]:
                    self.preprocessing[put_id]["object_scaler"][f"col_{i}"] = MinMaxScaler()
                    self.preprocessing[put_id]["object_scaler"][f"col_{i}"].fit(
                        array.iloc[:, [i]].values.reshape(-1, 1))

            if options["StandardScaler_cols"]:
                for i in options["StandardScaler_cols"]:
                    self.preprocessing[put_id]["object_scaler"][f"col_{i}"] = StandardScaler()
                    self.preprocessing[put_id]["object_scaler"][f"col_{i}"].fit(
                        array.iloc[:, [i]].values.reshape(-1, 1))

        else:
            self.preprocessing[put_id] = {'object_scaler': scaler}

    def create_tokenizer(self, put_id: int, text_list: list, **options):

        """

        Args:
            put_id: int
                Номер входа или выхода.
            text_list: list
                Список слов для обучения токенайзера.
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
        tokenizer = Tokenizer(**{'num_words': options['max_words_count'],
                                 'filters': options['filters'],
                                 'lower': False,
                                 'split': ' ',
                                 'char_level': False,
                                 'oov_token': '<UNK>'})
        tokenizer.fit_on_texts(text_list)

        self.preprocessing[put_id] = {'object_tokenizer': tokenizer}

    def create_word2vec(self, put_id: int, text_list: list, **options):

        """

        Args:
            put_id: int
                Номер входа или выхода.
            text_list: list
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
        text_list = [elem.split(' ') for elem in text_list]
        word2vec = Word2Vec(text_list, **{'size': options['word_to_vec_size'],
                                          'window': 10,
                                          'min_count': 1,
                                          'workers': 10,
                                          'iter': 10})

        self.preprocessing[put_id] = {'object_word2vec': word2vec}
