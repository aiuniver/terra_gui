import os
import joblib
import numpy as np
import imgaug.augmenters
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

        self.dataset_path = dataset_path
        self.preprocessing = {}

    # @staticmethod
    # def create_image_augmentation(options):
    #
    #     # КОСТЫЛЬ ИЗ-ЗА .NATIVE()
    #     for key, value in options.items():
    #         for name, elem in value.items():
    #             if key != 'ChannelShuffle':
    #                 if isinstance(options[key][name], list):
    #                     options[key][name] = tuple(options[key][name])
    #                 elif isinstance(options[key][name], dict):
    #                     for name2, elem2 in options[key][name].items():
    #                         options[key][name][name2] = tuple(options[key][name][name2])
    #
    #     aug_parameters = []
    #     for key, value in options.items():
    #         aug_parameters.append(getattr(imgaug.augmenters, key)(**value))
    #     augmentation = imgaug.augmenters.Sequential(aug_parameters, random_order=True)
    #
    #     return augmentation

    def load_preprocesses(self, put_data):

        for put in put_data.keys():
            self.preprocessing[put] = {}
            for col_name in put_data[put].keys():
                prep_path = os.path.join(self.dataset_path, 'preprocessing', str(put), f'{col_name}.gz')
                if os.path.isfile(prep_path):
                    preprocess_object = joblib.load(prep_path)
                    if repr(preprocess_object) in ['MinMaxScaler()', 'StandardScaler()']:
                        if 'clip' not in preprocess_object.__dict__.keys():
                            preprocess_object.clip = False
                    self.preprocessing[put].update([(col_name, preprocess_object)])
                else:
                    self.preprocessing[put].update([(col_name, None)])

    def create_scaler(self, **options):  # array=None,

        scaler = None
        if options.get('scaler') and options['scaler'] != 'no_scaler':
            if options['scaler'] == 'min_max_scaler':
                scaler = MinMaxScaler(feature_range=(options['min_scaler'], options['max_scaler']))
            elif options['scaler'] == 'standard_scaler':
                scaler = StandardScaler()
            elif options['scaler'] == 'terra_image_scaler':
                scaler = TerraImageScaler(shape=(options['height'], options['width']),
                                          min_max=(options['min_scaler'], options['max_scaler']))
        if not options['put'] in self.preprocessing.keys():
            self.preprocessing[options['put']] = {}
        self.preprocessing[options['put']].update([(options['cols_names'], scaler)])

    def create_tokenizer(self, text_list: list, **options):

        """

        Args:
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

        if not options['put'] in self.preprocessing.keys():
            self.preprocessing[options['put']] = {}
        self.preprocessing[options['put']].update([(options['cols_names'], tokenizer)])

    def create_word2vec(self, text_list: list, **options):

        """

        Args:
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

        if not options['put'] in self.preprocessing.keys():
            self.preprocessing[options['put']] = {}
        self.preprocessing[options['put']].update([(options['cols_names'], word2vec)])

    def inverse_data(self, options: dict):
        out_dict = {}
        for put_id, value in options.items():
            out_dict[put_id] = {}
            for col_name, array in value.items():
                if type(self.preprocessing[put_id][col_name]) == StandardScaler or \
                        type(self.preprocessing[put_id][col_name]) == MinMaxScaler:
                    out_dict[put_id].update({col_name: self.preprocessing[put_id][col_name].inverse_transform(array)})

                elif type(self.preprocessing[put_id][col_name]) == Tokenizer:
                    inv_tokenizer = {index: word for word, index in
                                     self.preprocessing[put_id][col_name].word_index.items()}
                    out_dict[put_id].update({col_name: ' '.join([inv_tokenizer[seq] for seq in array])})

                else:
                    out_dict[put_id].update({col_name: ' '.join(
                        [self.preprocessing[put_id][col_name].most_similar(
                            positive=[seq], topn=1)[0][0] for seq in array])})
        return out_dict
