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
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
    # def create_image(self, file_folder, image_path: str, **options):
    #
    #     shape = (options['height'], options['width'])
    #     # img = cv2.imread(os.path.join(self.file_folder, image_path)).reshape(*shape, 3)
    #     img = load_img(os.path.join(self.file_folder, image_path), target_size=shape)
    #     array = img_to_array(img, dtype=np.uint8)
    #     if options['net'] == 'Linear':
    #         array = array.reshape(np.prod(np.array(array.shape)))
    #     if options['put'] in self.augmentation.keys():
    #         if 'object_detection' in options.keys():
    #             txt_path = image_path[:image_path.rfind('.')] + '.txt'
    #             with open(os.path.join(self.file_folder, txt_path), 'r') as b_boxes:
    #                 bounding_boxes = b_boxes.read()
    #
    #             current_boxes = []
    #             for elem in bounding_boxes.split('\n'):
    #                 # b_box = self.yolo_to_imgaug(elem.split(' '), shape=array.shape[:2])
    #                 if elem:
    #                     b_box = elem.split(',')
    #                     b_box = [int(x) for x in b_box]
    #                     current_boxes.append(
    #                         BoundingBox(
    #                             **{'label': b_box[4], 'x1': b_box[0], 'y1': b_box[1], 'x2': b_box[2], 'y2': b_box[3]}))
    #
    #             bbs = BoundingBoxesOnImage(current_boxes, shape=array.shape)
    #             array, bbs_aug = self.augmentation[options['put']](image=array, bounding_boxes=bbs)
    #             list_of_bounding_boxes = []
    #             for elem in bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes:
    #                 bb = elem.__dict__
    #                 # b_box_coord = self.imgaug_to_yolo([bb['label'], bb['x1'], bb['y1'], bb['x2'], bb['y2']],
    #                 #                                   shape=array.shape[:2])
    #                 # if b_box_coord != ():
    #                 if bb:
    #                     list_of_bounding_boxes.append([bb['x1'], bb['y1'], bb['x2'], bb['y2'], bb['label']])
    #
    #             self.temporary['bounding_boxes'][txt_path] = list_of_bounding_boxes
    #         else:
    #             array = self.augmentation[options['put']](image=array)
    #
    #     array = array / 255
    #
    #     return array.astype('float32')

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
        if 'timeseries' in options.keys():
            lengh = options['lengh']
        else:
            lengh = 1
        row = self.df[list(range(row_number, row_number + lengh))].tolist()

        if 'StandardScaler' in options.values() or 'MinMaxScaler' in options.values():
            array = self.scaler[options['put']].transform(row)
        else:
            if 'StandardScaler' in options.keys():
                for i in options['StandardScaler']:
                    for j in range(lengh):
                        row[j][i] = self.scaler[options['put']]['StandardScaler'].transform(
                            np.array(row[j][i]).reshape(-1, 1)).tolist()

            if 'MinMaxScaler' in options.keys():
                for i in options['MinMaxScaler']:
                    for j in range(lengh):
                        row[j][i] = self.scaler[options['put']]['MinMaxScaler'].transform(
                            np.array(row[j][i]).reshape(-1, 1)).tolist()

            if 'Categorical' in options.keys():
                for i in options['Categorical']['lst_cols']:
                    for j in range(lengh):
                        row[j][i] = list(options['Categorical'][f'col_{i}']).index(row[j][i])

            if 'Categorical_ranges' in options.keys():
                for i in options['Categorical_ranges']['lst_cols']:
                    for j in range(lengh):
                        for k in range(len(options['Categorical_ranges'][f'col_{i}'])):
                            if row[j][i] <= options['Categorical_ranges'][f'col_{i}'][f'range_{k}']:
                                row[j][i] = k
                                break

            if 'one_hot_encoding' in options.keys():
                for i in options['one_hot_encoding']['lst_cols']:
                    for j in range(lengh):
                        row[j][i] = utils.to_categorical(row[j][i], options['one_hot_encoding'][f'col_{i}'],
                                                         dtype='uint8').tolist()

            array = []
            for i in row:
                tmp = []
                for j in i:
                    if type(j) == list:
                        if type(j[0]) == list:
                            tmp.extend(j[0])
                        else:
                            tmp.extend(j)
                    else:
                        tmp.append(j)
                array.append(tmp)

        array = np.array(array)

        return array

    def create_classification(self, file_folder, index, **options):

        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        index = np.array(index)

        return index

    def create_regression(self, file_folder, index, **options):

        if 'scaler' in options.keys():
            index = self.scaler[options['put']].transform(np.array(index).reshape(-1, 1)).reshape(1, )[0]
        array = np.array(index)

        return array

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

    def create_timeseries(self, row_number, **options):
        """
            Args:
                row_number: номер строки с сырыми данными для предсказания значения,
                **options: Параметры обработки колонок:
                    depth: количество значений для предсказания
                    lengh: количество примеров для обучения
                    put: str  Индекс входа или выхода.
            Returns:
                array: np.ndarray
                    Массив обработанных данных.
        """

        array = self.y_subdf[list(range(
            row_number + options['lengh'], row_number + options['lengh'] + options['depth']))]

        if 'StandardScaler' in options.values() or 'MinMaxScaler' in options.values():
            array = self.scaler[options['put']].transform(array)

        return array

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
