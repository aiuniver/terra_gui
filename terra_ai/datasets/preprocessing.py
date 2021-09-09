import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gensim.models.word2vec import Word2Vec
from terra_ai.data.datasets.extra import LayerScalerImageChoice
from tensorflow.keras.preprocessing.text import Tokenizer


class CreatePreprocessing(object):

    def __init__(self):

        self.preprocessing = {}

    def load_preprocesses(self, path_dataset, keys):

        for key in keys:
            preprocess = {}
            for param in ['augmentation', 'scaler', 'tokenizer', 'word2vec']:
                if os.path.isfile(os.path.join(path_dataset, param, f'{key}.gz')):
                    preprocess[param] = joblib.load(os.path.join(path_dataset, param, f'{key}.gz'))
                else:
                    preprocess[param] = None
            self.preprocessing[key] = preprocess
        pass

    def create_dull(self, put_id: int):

        self.preprocessing[put_id] = {'dull': None}

    def create_scaler(self, put_id: int, array=None, **options):

        scaler = None
        if options['scaler'] == LayerScalerImageChoice.min_max_scaler:
            scaler = MinMaxScaler()
            array = array.reshape(-1, 1) if array else np.array([[0], [255]])
            scaler.fit(array)

        self.preprocessing[put_id] = {'object_scaler': scaler}

    def create_tokenizer(self, put_id: int, text_list: list, **options):

        """

        Args:
            put_id: int
                Номер входа или выхода.
            text_list: list
                Список слов для обучения Word2Vec.
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
