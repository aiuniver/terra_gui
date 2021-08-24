import os
import cv2
import random
import numpy as np

from sklearn.cluster import KMeans
from gensim.models.word2vec import Word2Vec

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from librosa import load as librosa_load
import librosa.feature as librosa_feature
from pydantic.color import Color

from ..data.datasets.extra import (
    LayerNetChoice,
    LayerVideoFrameModeChoice,
    LayerVideoFillModeChoice,
)

from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils


class CreateArray(object):
    def __init__(self):

        self.scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.augmentation: dict = {}
        self.temporary: dict = {"bounding_boxes": {}}

        self.file_folder = None
        self.txt_list: dict = {}

    def create_image(self, file_folder: str, image_path: str, **options):

        shape = (options["height"], options["width"])
        # array = cv2.imread(os.path.join(file_folder, image_path)).reshape((shape[0], shape[1], 3)).astype('uint8')
        img = load_img(os.path.join(file_folder, image_path), target_size=shape)
        array = img_to_array(img, dtype=np.uint8)
        if options["net"] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options["put"] in self.augmentation.keys():
            if "object_detection" in options.keys():
                txt_path = image_path[: image_path.rfind(".")] + ".txt"
                with open(os.path.join(file_folder, txt_path), "r") as b_boxes:
                    bounding_boxes = b_boxes.read()

                current_boxes = []
                for elem in bounding_boxes.split("\n"):
                    # b_box = self.yolo_to_imgaug(elem.split(' '), shape=array.shape[:2])
                    if elem:
                        b_box = elem.split(",")
                        b_box = [int(x) for x in b_box]
                        current_boxes.append(
                            BoundingBox(
                                **{
                                    "label": b_box[4],
                                    "x1": b_box[0],
                                    "y1": b_box[1],
                                    "x2": b_box[2],
                                    "y2": b_box[3],
                                }
                            )
                        )

                bbs = BoundingBoxesOnImage(current_boxes, shape=array.shape)
                array, bbs_aug = self.augmentation[options["put"]](
                    image=array, bounding_boxes=bbs
                )
                list_of_bounding_boxes = []
                for elem in (
                    bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes
                ):
                    bb = elem.__dict__
                    # b_box_coord = self.imgaug_to_yolo([bb['label'], bb['x1'], bb['y1'], bb['x2'], bb['y2']],
                    #                                   shape=array.shape[:2])
                    # if b_box_coord != ():
                    if bb:
                        list_of_bounding_boxes.append(
                            [bb["x1"], bb["y1"], bb["x2"], bb["y2"], bb["label"]]
                        )

                self.temporary["bounding_boxes"][txt_path] = list_of_bounding_boxes
            else:
                array = self.augmentation[options["put"]](image=array)

        array = array / 255

        return array.astype("float32")

    def create_video(
        self, file_folder: str, video: str, slicing: list, **options
    ) -> np.ndarray:

        """
        Args:
            file_folder: str
                Путь к папке.
            video: str
                Путь к файлу.
            slicing: list
                [начало: int, конец: int].
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

            if frame_mode == LayerVideoFrameModeChoice.stretch:
                resized = resize_layer(one_frame[None, ...])
                resized = resized.numpy().squeeze().astype("uint8")
            elif frame_mode == LayerVideoFrameModeChoice.keep_proportions:
                # height
                resized = one_frame.copy()
                if original_shape[0] > target_shape[0]:
                    resized = resized[
                        int(original_shape[0] / 2 - target_shape[0] / 2) : int(
                            original_shape[0] / 2 - target_shape[0] / 2
                        )
                        + target_shape[0],
                        :,
                    ]
                else:
                    black_bar = np.zeros(
                        (
                            int((target_shape[0] - original_shape[0]) / 2),
                            original_shape[1],
                            3,
                        ),
                        dtype="uint8",
                    )
                    resized = np.concatenate((black_bar, resized))
                    resized = np.concatenate((resized, black_bar))
                # width
                if original_shape[1] > target_shape[1]:
                    resized = resized[
                        :,
                        int(original_shape[1] / 2 - target_shape[1] / 2) : int(
                            original_shape[1] / 2 - target_shape[1] / 2
                        )
                        + target_shape[1],
                    ]
                else:
                    black_bar = np.zeros(
                        (
                            target_shape[0],
                            int((target_shape[1] - original_shape[1]) / 2),
                            3,
                        ),
                        dtype="uint8",
                    )
                    resized = np.concatenate((black_bar, resized), axis=1)
                    resized = np.concatenate((resized, black_bar), axis=1)

            return resized

        def add_frames(video_array, fill_mode, frames_to_add, total_frames):

            frames: np.ndarray = np.array([])

            if fill_mode == LayerVideoFillModeChoice.black_frames:
                frames = np.zeros((frames_to_add, *shape, 3), dtype="uint8")
            elif fill_mode == LayerVideoFillModeChoice.average_value:
                mean = np.mean(video_array, axis=0, dtype="uint16")
                frames = np.full((frames_to_add, *mean.shape), mean, dtype="uint8")
            elif fill_mode == LayerVideoFillModeChoice.last_frames:
                if total_frames > frames_to_add:
                    frames = np.flip(video_array[-frames_to_add:], axis=0)
                elif total_frames <= frames_to_add:
                    for i in range(frames_to_add // total_frames):
                        frames = np.flip(video_array[-total_frames:], axis=0)
                        video_array = np.concatenate((video_array, frames), axis=0)
                    if frames_to_add + total_frames != video_array.shape[0]:
                        frames = np.flip(
                            video_array[
                                -(frames_to_add + total_frames - video_array.shape[0]) :
                            ],
                            axis=0,
                        )
            video_array = np.concatenate((video_array, frames), axis=0)

            return video_array

        array = []
        shape = (options["height"], options["width"])
        frames_count = slicing[1] - slicing[0]
        resize_layer = Resizing(*shape)

        cap = cv2.VideoCapture(os.path.join(file_folder, video))
        width = int(cap.get(3))
        height = int(cap.get(4))
        max_frames = int(cap.get(7))
        cap.set(1, slicing[0])
        try:
            for _ in range(frames_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if shape != (height, width):
                    frame = resize_frame(
                        one_frame=frame,
                        original_shape=(height, width),
                        target_shape=shape,
                        frame_mode=options["frame_mode"],
                    )
                frame = frame[:, :, [2, 1, 0]]
                array.append(frame)
        finally:
            cap.release()

        array = np.array(array)
        if max_frames < frames_count:
            array = add_frames(
                video_array=array,
                fill_mode=options["fill_mode"],
                frames_to_add=frames_count - max_frames,
                total_frames=max_frames,
            )

        return array

    def create_text(self, _, text: str, slicing: list, **options):

        """
        Args:
            _: None
                Путь к файлу.
            text: str
                Отрывок текста.
            slicing: list
                [начало: int, конец: int].
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
        text = text.split(" ")
        if slicing[1] - slicing[0] < len(text):
            text = text[slicing[0] : slicing[1]]

        if options["embedding"]:
            array = self.tokenizer[options["put"]].texts_to_sequences([text])[0]
        elif options["bag_of_words"]:
            array = self.tokenizer[options["put"]].texts_to_matrix([text])[0]
        elif options["word_to_vec"]:
            for word in text:
                array.append(self.word2vec[options["put"]][word])

        if len(array) < slicing[1] - slicing[0]:
            words_to_add = [0 for _ in range((slicing[1] - slicing[0]) - len(array))]
            array += words_to_add

        array = np.array(array)

        return array

    def create_audio(
        self, file_folder: str, audio: str, slicing: list, **options
    ) -> np.ndarray:

        array = []
        parameter = options["parameter"]
        sample_rate = options["sample_rate"]
        y, sr = librosa_load(
            path=os.path.join(file_folder, audio),
            sr=options.get("sample_rate"),
            offset=slicing[0],
            duration=slicing[1] - slicing[0],
            res_type="kaiser_best",
        )
        if sample_rate > len(y):
            zeros = np.zeros((sample_rate - len(y),))
            y = np.concatenate((y, zeros))

        if parameter in [
            "chroma_stft",
            "mfcc",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
        ]:
            array = getattr(librosa_feature, parameter)(y=y, sr=sr)
        elif parameter == "rms":
            array = getattr(librosa_feature, parameter)(y=y)[0]
        elif parameter == "zero_crossing_rate":
            array = getattr(librosa_feature, parameter)(y=y)
        elif parameter == "audio_signal":
            array = y

        array = np.array(array)
        if array.dtype == "float64":
            array = array.astype("float32")

        return array

    def create_dataframe(self, _, row_number: int, **options):
        """
        Args:
            _: путь к файлу,
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
        if "timeseries" in options.keys():
            length = options["length"]
        else:
            length = 1
        row_number = int(row_number)
        row = self.df_ts.iloc[
            list(range(row_number, row_number + length)), list(range(len(self.columns)))
        ].values.tolist()
        if options["xlen_step"]:
            row = row[0]
        if (
            "standard_scaler" in options.values()
            or "min_max_scaler" in options.values()
        ):
            array = self.scaler[options["put"]].transform(row)
        else:
            if "MinMaxScaler" in options.keys():
                for j in range(length):
                    for i in options["MinMaxScaler"]:
                        row[j][i] = (
                            self.scaler[options["put"]]["MinMaxScaler"][f"col_{i + 1}"]
                            .transform(np.array(row[j][i]).reshape(-1, 1))
                            .tolist()
                        )

            if "StandardScaler" in options.keys():
                for j in range(length):
                    for i in options["StandardScaler"]:
                        row[j][i] = (
                            self.scaler[options["put"]]["StandardScaler"][
                                f"col_{i + 1}"
                            ]
                            .transform(np.array(row[j][i]).reshape(-1, 1))
                            .tolist()
                        )

            if "Categorical" in options.keys():
                for j in range(length):
                    for i in options["Categorical"]["lst_cols"]:
                        row[j][i] = list(options["Categorical"][f"col_{i}"]).index(
                            row[j][i]
                        )

            if "Categorical_ranges" in options.keys():
                for j in range(length):
                    for i in options["Categorical_ranges"]["lst_cols"]:
                        for k in range(len(options["Categorical_ranges"][f"col_{i}"])):
                            if (
                                row[j][i]
                                <= options["Categorical_ranges"][f"col_{i}"][
                                    f"range_{k}"
                                ]
                            ):
                                row[j][i] = k
                                break

            if "one_hot_encoding" in options.keys():
                for j in range(length):
                    for i in options["one_hot_encoding"]["lst_cols"]:
                        row[j][i] = utils.to_categorical(
                            row[j][i],
                            options["one_hot_encoding"][f"col_{i}"],
                            dtype="uint8",
                        ).tolist()

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

    @staticmethod
    def create_classification(_, class_name, **options):

        index = options["classes_names"].index(class_name)
        if options["one_hot_encoding"]:
            index = utils.to_categorical(
                index, num_classes=options["num_classes"], dtype="uint8"
            )
        index = np.array(index)

        return index

    def create_regression(self, _, index, **options):
        if (
            "standard_scaler" in options.values()
            or "min_max_scaler" in options.values()
        ):
            index = (
                self.scaler[options["put"]]
                .transform(np.array(index).reshape(-1, 1))
                .reshape(
                    1,
                )[0]
            )
        array = np.array(index)
        return array

    @staticmethod
    def create_segmentation(
        file_folder: str, image_path: str, **options: dict
    ) -> np.ndarray:

        """

        Args:
            file_folder: str
                Путь к папке
                Путь к файлу
            image_path: str
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
            km = KMeans(n_clusters=options["num_classes"])
            km.fit(mask_image)
            labels = km.labels_
            cl_cent = km.cluster_centers_.astype("uint8")[: max(labels) + 1]
            cl_mask = utils.to_categorical(labels, max(labels) + 1, dtype="uint8")
            cl_mask = cl_mask.reshape(
                options["shape"][0], options["shape"][1], cl_mask.shape[-1]
            )
            mask_ohe = np.zeros(options["shape"])
            for k, color in enumerate(options["classes_colors"]):
                rgb = Color(color).as_rgb_tuple()
                mask = np.zeros(options["shape"])

                for j, cl_rgb in enumerate(cl_cent):
                    if (
                        rgb[0]
                        in range(
                            cl_rgb[0] - options["mask_range"],
                            cl_rgb[0] + options["mask_range"],
                        )
                        and rgb[1]
                        in range(
                            cl_rgb[1] - options["mask_range"],
                            cl_rgb[1] + options["mask_range"],
                        )
                        and rgb[2]
                        in range(
                            cl_rgb[2] - options["mask_range"],
                            cl_rgb[2] + options["mask_range"],
                        )
                    ):
                        mask = cl_mask[:, :, j]

                if k == 0:
                    mask_ohe = mask
                else:
                    mask_ohe = np.dstack((mask_ohe, mask))

            return mask_ohe

        img = load_img(
            path=os.path.join(file_folder, image_path), target_size=options["shape"]
        )
        array = img_to_array(img, dtype=np.uint8)
        array = cluster_to_ohe(array)

        return array

    def create_text_segmentation(self, _, text: str, slicing: list, **options):

        array = []
        if slicing[1] - slicing[0] < len(text):
            text = text[slicing[0] : slicing[1]]

        for elem in text:
            tags = [0 for _ in range(options["num_classes"])]
            if elem:
                for idx in elem:
                    tags[idx] = 1
            array.append(tags)
        array = np.array(array, dtype="uint8")

        return array

    def create_timeseries(self, _, row_number, **options):
        """
        Args:
            _: путь к файлу,
            row_number: номер строки с сырыми данными для предсказания значения,
            **options: Параметры обработки колонок:
                depth: количество значений для предсказания
                length: количество примеров для обучения
                put: str  Индекс входа или выхода.
        Returns:
            array: np.ndarray
                Массив обработанных данных.
        """

        if options["bool_trend"]:
            array = np.array(row_number)

        else:
            row_number = int(row_number)
            array = self.df_ts.loc[
                list(
                    range(
                        row_number + options["length"],
                        row_number + options["length"] + options["depth"],
                    )
                ),
                list(self.y_cols),
            ].values

            if (
                "standard_scaler" in options.values()
                or "min_max_scaler" in options.values()
            ):
                array = self.scaler[options["put"]].transform(array)

        return array

    def create_object_detection(self, file_folder: str, txt_path: str, **options):

        """
        Args:
            file_folder: str
                Путь к файлу
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

            boxes1 = tf_concat(
                [
                    boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                    boxes1[..., :2] + boxes1[..., 2:] * 0.5,
                ],
                axis=-1,
            )
            boxes2 = tf_concat(
                [
                    boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                    boxes2[..., :2] + boxes2[..., 2:] * 0.5,
                ],
                axis=-1,
            )

            left_up = tf_maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = tf_minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = tf_maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area

            return 1.0 * inter_area / union_area

        # height: int = options['height']
        # width: int = options['width']
        num_classes: int = options["num_classes"]
        zero_boxes_flag: bool = False
        strides = np.array([8, 16, 32])
        output_levels = len(strides)
        train_input_sizes = 416
        anchor_per_scale = 3
        yolo_anchors = [
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]],
        ]
        anchors = (np.array(yolo_anchors).T / strides).T
        max_bbox_per_scale = 100
        train_input_size = random.choice([train_input_sizes])
        train_output_sizes = train_input_size // strides

        if self.temporary["bounding_boxes"]:
            real_boxes = self.temporary["bounding_boxes"][txt_path]
        else:
            with open(os.path.join(file_folder, txt_path), "r") as txt:
                bb_file = txt.read()
            real_boxes = []
            for elem in bb_file.split("\n"):
                tmp = []
                if elem:
                    for num in elem.split(","):
                        tmp.append(int(num))
                    real_boxes.append(tmp)

        if not real_boxes:
            zero_boxes_flag = True
            real_boxes = [[0, 0, 0, 0, 0]]
        real_boxes = np.array(real_boxes)
        label = [
            np.zeros(
                (
                    train_output_sizes[i],
                    train_output_sizes[i],
                    anchor_per_scale,
                    5 + num_classes,
                )
            )
            for i in range(output_levels)
        ]
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

            bbox_xywh = np.concatenate(
                [
                    (bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                    bbox_coordinate[2:] - bbox_coordinate[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(output_levels):  # range(3):
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = (
                        0.0 if zero_boxes_flag else 1.0
                    )
                    label[i][yind, xind, iou_mask, 5:] = smooth_one_hot

                    bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchor_per_scale)
                best_anchor = int(best_anchor_ind % anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(
                    np.int32
                )

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = (
                    0.0 if zero_boxes_flag else 1.0
                )
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return (
            np.array(label_sbbox, dtype="float32"),
            np.array(sbboxes, dtype="float32"),
            np.array(label_mbbox, dtype="float32"),
            np.array(mbboxes, dtype="float32"),
            np.array(label_lbbox, dtype="float32"),
            np.array(lbboxes, dtype="float32"),
        )

    def create_scaler(self):

        pass

    def create_tokenizer(self, put_id: int, **options):

        """

        Args:
            put_id: int
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

        self.tokenizer[put_id] = Tokenizer(**options)

        pass

    def create_word2vec(self, put_id: int, words: list, **options) -> None:

        """

        Args:
            put_id: int
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

        self.word2vec[put_id] = Word2Vec(words, **options)

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
                if attr == "tokenizer":
                    if array.shape[0] == self.tokenizer[put].num_words:
                        idx = 0
                        arr = []
                        for num in array:
                            if num == 1:
                                arr.append(idx)
                            idx += 1
                        array = np.array(arr)
                    inv_tokenizer = {
                        index: word
                        for word, index in self.tokenizer[put].word_index.items()
                    }
                    inverted_data = " ".join([inv_tokenizer[seq] for seq in array])

                elif attr == "word2vec":
                    text_list = []
                    for i in range(len(array)):
                        text_list.append(
                            self.word2vec[put].wv.most_similar(
                                positive=np.expand_dims(array[i], axis=0), topn=1
                            )[0][0]
                        )
                    inverted_data = " ".join(text_list)

                elif attr == "scaler":
                    original_shape = array.shape
                    array = array.reshape(-1, 1)
                    array = self.scaler[put].inverse_transform(array)
                    inverted_data = array.reshape(original_shape)
            break

        return inverted_data
