import os
import cv2
import numpy as np

from sklearn.cluster import KMeans
from gensim.models.word2vec import Word2Vec
from tqdm.notebook import tqdm

from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
from librosa import load as librosa_load
import librosa.feature as librosa_feature


class CreateArray(object):

    def __init__(self):

        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.augmentation: dict = {}

        self.file_folder = None
        self.txt_list: dict = {}

    def create_image(self, file_folder, image_path: str, **options):

        shape = (options['height'], options['width'])
        img = load_img(path=os.path.join(file_folder, image_path), target_size=shape)
        array = img_to_array(img, dtype=np.uint8)
        if options['net'] == 'Linear':
            array = array.reshape(np.prod(np.array(array.shape)))

        return array

    def create_video(self, video_path, **options) -> np.ndarray:

        """

        Args:
            video_path: str
                Путь к файлу
            **options: Параметры сегментации:
                height: int
                    Высота кадра.
                width: int
                    Ширина кадра.
                max_frames: int
                    Максимальное количество кадров.
                mode: str
                    Режим обработки кадра (Сохранить пропорции, Растянуть).
                x_len: int
                    Длина окна выборки.
                step: int
                    Шаг окна выборки.

        Returns:
            array: np.ndarray
                Массив видео.

        """

        def resize_frame(one_frame, original_shape, target_shape, mode):

            resized = None

            if mode == 'Растянуть':
                resized = resize_layer(one_frame[None, ...])
                resized = resized.numpy().squeeze().astype('uint8')
            elif mode == 'Сохранить пропорции':
                # height
                resized = one_frame.copy()
                if original_shape[0] > target_shape[0]:
                    resized = resized[int(original_shape[0] / 2 - target_shape[0] / 2):int(original_shape[0] / 2 - target_shape[0] / 2) + target_shape[0], :]
                else:
                    black_bar = np.zeros((int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3), dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    resized = np.concatenate((resized, black_bar))
                # width
                if original_shape[1] > target_shape[1]:
                    resized = resized[:, int(original_shape[1] / 2 - target_shape[1] / 2):int(original_shape[1] / 2 - target_shape[1] / 2) + target_shape[1]]
                else:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3), dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    resized = np.concatenate((resized, black_bar), axis=1)

            # resized = resized.numpy().squeeze()

            return resized

        array = []
        shape = (options['height'], options['width'])
        resize_layer = Resizing(*shape)

        cap = cv2.VideoCapture(os.path.join(self.file_folder, video_path))
        height = int(cap.get(4))
        width = int(cap.get(3))
        # fps = int(cap.get(5))
        frame_count = int(cap.get(7))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if shape != (height, width):
                    frame = resize_frame(frame, (height, width), shape, options['mode'])
                frame = frame[:, :, [2, 1, 0]]
                array.append(frame)
                if len(array) == options['max_frames']:
                    break
        finally:
            cap.release()

        array = np.array(array)
        if frame_count < options['max_frames']:
            add_frames = np.zeros((options['max_frames'] - frame_count, *shape, 3), dtype='uint8')
            array = np.concatenate((array, add_frames), axis=0)

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

        array = np.array(array)

        return array

    def create_audio(self, sample: dict, **options):

        array = []

        [[filepath, slicing]] = sample.items()
        y, sr = librosa_load(path=os.path.join(self.file_folder, filepath), sr=options.get('sample_rate'),
                             offset=slicing[0], duration=slicing[1] - slicing[0], res_type='kaiser_best')

        for feature in options.get('features', []):
            if feature in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                array.append(getattr(librosa_feature, feature)(y=y, sr=sr))
            elif feature == 'rms':
                array.append(getattr(librosa_feature, feature)(y=y)[0])
            elif feature == 'zero_crossing_rate':
                array.append(getattr(librosa_feature, feature)(y=y))
            elif feature == 'audio_signal':
                array.append(y)

        return tuple(array)

    def create_dataframe(self):

        pass

    def create_classification(self, index, **options):

        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        index = np.array(index)

        return index

    def create_regression(self):

        pass

    def create_segmentation(self, file_folder, image_path: str, **options: dict) -> np.ndarray:

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
                # rgb = rgb.as_rgb_tuple()
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

        img = load_img(path=os.path.join(file_folder, image_path), target_size=options['shape'])
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
                height: int
                    Высота изображения.
                width: int
                    Ширина изображения.
                num_classes: tuple
                    Количество классов.

        Returns:
            array: np.ndarray
                Массивы в трёх выходах.

        """

        height: int = options['height']
        width: int = options['width']
        num_classes: int = options['num_classes']

        with open(os.path.join(self.file_folder, txt_path), 'r') as txt:
            bb_file = txt.read()
        real_boxes = []
        for elem in bb_file.split('\n'):
            tmp = []
            if elem:
                for num in elem.split(' '):
                    tmp.append(float(num))
                real_boxes.append(tmp)
        real_boxes = np.array(real_boxes)
        real_boxes = real_boxes[:, [1, 2, 3, 4, 0]]
        anchors = np.array(
            [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        num_layers = 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        real_boxes = np.array(real_boxes, dtype='float32')
        input_shape = np.array((height, width), dtype='int32')

        boxes_wh = real_boxes[..., 2:4] * input_shape

        cells = [13, 26, 52]
        y_true = [np.zeros((cells[n], cells[n], len(anchor_mask[n]), 5 + num_classes), dtype='float32') for n in
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

        return np.array(y_true[0]), np.array(y_true[1]), np.array(y_true[2])

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
