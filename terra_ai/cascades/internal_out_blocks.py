import colorsys
import os.path
import random
import uuid

import numpy as np
import moviepy.editor as moviepy_editor

from pydantic.color import Color
from typing import Any
from io import BytesIO
from pydub import AudioSegment
from tensorflow.keras.utils import save_img
from PIL import Image

from .common import is_terra_file
from .main_blocks import BaseBlock, CascadeBlock


class EmptyOut(BaseBlock):

    front_name = 'Проброс входа'
    data_type = 'initial'

    def execute(self, **kwargs) -> Any:
        return kwargs.get('source')


class ArrayOut(BaseBlock):

    front_name = 'Входной массив'
    data_type = 'array'

    def execute(self, **kwargs) -> np.ndarray:
        return kwargs.get('array')


class ImageArrayOut(ArrayOut):
    data_type = 'image_array'


class InputImageArrayOut(ImageArrayOut):
    data_type = 'image_array'

    def execute(self, **kwargs) -> np.ndarray:
        return kwargs.get('image_array')


class BboxesArrayOut(ArrayOut):
    data_type = 'bboxes'

    def execute(self, **kwargs) -> np.ndarray:
        return kwargs.get('bboxes')


class AudioArrayOut(ArrayOut):
    data_type = 'audio_array'


class TextArrayOut(ArrayOut):
    data_type = 'text_array'


class VideoArrayOut(ArrayOut):
    data_type = 'video_array'


class NativeOut(BaseBlock):

    front_name = 'Результат предикта'
    data_type = 'classes'

    def execute(self, **kwargs) -> np.ndarray:
        return kwargs.get('model_predict')


class SegmentationNativeOut(NativeOut):
    data_type = 'mask_colors'


class TextSegmentationNativeOut(NativeOut):
    data_type = 'tags_colors'


class YoloNativeOut(NativeOut):
    data_type = 'bboxes'

    def execute(self, **kwargs) -> tuple:
        return kwargs.get('model_predict'), kwargs.get('classes')


class ClassificationOut(BaseBlock):

    front_name = 'Список классов с вероятностями'
    data_type = 'classes_list'

    @staticmethod
    def sort_dict(dict_to_sort: dict):
        sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)

    def get_out(self, model_predict, classes):
        labels_from_array = []
        for class_idx in model_predict:
            class_dict = {}
            for i, cl in enumerate(classes):
                class_dict[cl] = class_idx[i]
            sorted_class = self.sort_dict(dict_to_sort=class_dict)

            labels_dist = []
            for j in range(len(classes)):
                labels_dist.append((sorted_class[0][j], round(float(sorted_class[1][j]) * 100, 1)))
            labels_from_array.append(labels_dist)
        return labels_from_array

    def execute(self, **kwargs) -> list:
        classes = []
        model_predict = kwargs.get('model_predict', [])
        options = kwargs.get('options')

        for block, params in options.get('outputs', {}).items():
            if params.get('task', '').lower() == 'classification':
                classes = params.get('classes_names')

        output = []
        if isinstance(model_predict, list):
            for arr_ in model_predict:
                output.append(self.get_out(model_predict=arr_, classes=classes))
        else:
            output = self.get_out(model_predict=model_predict, classes=classes)

        return output


class RGBMaskOut(BaseBlock):

    front_name = 'Изображение маски в RGB'
    data_type = 'image_mask'

    def execute(self, **kwargs) -> np.ndarray:
        model_predict = kwargs.get('model_predict')
        options = list(kwargs.get('options').get('outputs').values())[0]

        array = np.expand_dims(np.argmax(model_predict, axis=-1), axis=-1) * 512
        for i, color in enumerate(options.get('classes_colors')):
            array = np.where(array == i * 512, np.array(Color(color).as_rgb_tuple()), array)
        return array.astype("uint8")


class ImageSegmentationOut(BaseBlock):

    front_name = 'Список классов с цветами'
    data_type = 'classes_list'

    def execute(self, **kwargs):
        model_predict = kwargs.get('model_predict')
        options = kwargs.get('options')

        names, colors = self.get_params(**options)
        sum_list = [np.sum(model_predict[:, :, :, i]) for i in range(model_predict.shape[-1])]
        return [(names[i], colors[i]) for i, count in enumerate(sum_list) if count > 0]

    @staticmethod
    def get_params(**options):

        names = options["outputs"]["2"]["classes_names"]
        colors = [Color(i).as_rgb_tuple() for i in options["outputs"]["2"]["classes_colors"]]
        return names, colors


class TextSegmentationOut(BaseBlock):

    front_name = 'Размеченный текст'
    data_type = 'colored_text'

    def execute(self, **kwargs) -> tuple:
        source = kwargs.get('source')
        model_predict = kwargs.get('model_predict')
        options = kwargs.get('options')

        dataset_tags, classes_names, colors = self.get_params(**options)

        text_segmentation = self.text_colorization(
            text=source.get('1').get('1_text')[0], label_array=model_predict[0], tag_list=dataset_tags
        )
        data = [('<p1>', '<p1>', (200, 200, 200))]
        for tag in colors.keys():
            data.append(
                (tag, classes_names[tag], colors[tag])
            )
        return text_segmentation, data

    def text_colorization(self, text: str, label_array: np.ndarray, tag_list: list):
        text = text.split(" ")
        labels = self.reformat_tags(label_array, tag_list)
        colored_text = []
        for w, word in enumerate(text):
            colored_text.append(self.add_tags_to_word(word, labels[w]))
        return ' '.join(colored_text)

    @staticmethod
    def add_tags_to_word(word: str, tag_: str):
        if tag_:
            for t in tag_:
                word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
            return word
        else:
            return f"<p1>{word}</p1>"

    @staticmethod
    def reformat_tags(y_array: np.ndarray, tag_list: list, sensitivity: float = 0.9):
        norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
        reformat_list = []
        for word_tag in norm_array:
            if np.sum(word_tag) == 0:
                reformat_list.append(None)
            else:
                mix_tag = []
                for wt, wtag in enumerate(word_tag):
                    if wtag == 1:
                        mix_tag.append(tag_list[wt])
                reformat_list.append(mix_tag)
        return reformat_list

    @staticmethod
    def get_params(**options):
        classes_colors = {}
        classes_names = {}

        dataset_tags = options.get('instructions').get('2').get('2_text_segmentation').get("open_tags").split()
        names = options["outputs"]["2"]["classes_names"]
        colors = options["outputs"]["2"]["classes_colors"]

        if colors:
            for i, name in enumerate(dataset_tags):
                classes_colors[name] = colors[i].as_rgb_tuple()
                classes_names[name] = names[i]
        else:
            hsv_tuples = [(x / len(dataset_tags), 1., 1.) for x in range(len(dataset_tags))]
            gen_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            gen_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), gen_colors))
            for i, name in enumerate(dataset_tags):
                classes_colors[name] = gen_colors[i]
                classes_names[name] = names[i]

        return dataset_tags, classes_names, classes_colors


class SaveAudioOut(BaseBlock):

    front_name = 'Аудиофайл'
    file_type = 'webm'
    data_type = 'initial_file'

    def execute(self, **kwargs):
        source = kwargs.get('initial').get('1').get('1_audio')[0]
        save_path = kwargs.get('save_path')

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"result_{uuid.uuid4()}.{self.file_type}"
        )

        AudioSegment.from_file(source).export(path_, format=f"{self.file_type}")

        return path_


class SaveImgOut(EmptyOut):
    front_name = 'Файл изображения'
    file_type = 'webp'
    data_type = 'initial_file'

    def execute(self, **kwargs):
        result = super().execute(**kwargs).get('1').get('1_image')[0]
        save_path = kwargs.get('save_path')

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"initial_{uuid.uuid4()}.{self.file_type}"
        )
        img = Image.open(result)
        img.save(path_, f"{self.file_type}")

        return path_


class SaveVideoOut(EmptyOut):
    front_name = 'Файл видео'
    file_type = 'webm'
    data_type = 'initial_file'

    def execute(self, **kwargs):
        save_path = kwargs.get('save_path')
        source = kwargs.get('source').get('1').get('1_video')[0]

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"initial_{uuid.uuid4()}.{self.file_type}"
        )
        clip = moviepy_editor.VideoFileClip(source)
        clip.write_videofile(path_, preset='ultrafast', bitrate='3000k')

        return path_


class SaveTextOut(EmptyOut):
    front_name = 'Текстовый файл'
    file_type = 'txt'
    data_type = 'initial_file'

    def execute(self, **kwargs):
        result = super().execute(**kwargs).get('1').get('1_text')[0]
        save_path = kwargs.get('save_path')

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"initial_{uuid.uuid4()}.{self.file_type}"
        )
        with open(path_, 'w', encoding='utf-8') as text:
            text.write(result)
        return path_


class SaveTextSegmentationOut(TextSegmentationOut):
    front_name = 'Сохранение сегментированного текста'
    file_type = 'txt'
    data_type = 'deploy'

    def execute(self, **kwargs):
        result = super().execute(**kwargs)
        save_path = kwargs.get('save_path')

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"result_{uuid.uuid4()}.{self.file_type}"
        )
        with open(path_, 'w', encoding='utf-8') as text:
            text.write(result[0])
        return path_


class SaveImageSegmentOut(RGBMaskOut):

    front_name = 'Файл маски'
    file_type = 'webp'
    data_type = 'deploy'

    def execute(self, **kwargs):
        result = super().execute(**kwargs)
        while len(result.shape) != 3:
            result = result[0]

        save_path = kwargs.get('save_path')

        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"result_{uuid.uuid4()}.{self.file_type}"
        )
        save_img(path_, result)

        return path_


class SaveImgFromArrayOut(SaveImgOut):

    front_name = 'Файл изображения из массива'
    data_type = 'deploy_file'

    def execute(self, **kwargs):
        image = kwargs.get('model_predict')
        save_path = kwargs.get('save_path')
        examples = kwargs.get('examples')
        if image.any():
            if not os.path.exists(save_path) and not is_terra_file(save_path):
                os.makedirs(save_path, exist_ok=True)

            path_ = save_path if is_terra_file(save_path) else os.path.join(
                save_path, f"result_{uuid.uuid4()}.{self.file_type}"
            )
            if examples:
                try:
                    print(image[random.randint(0, examples)].shape, '\n', image[random.randint(0, examples)])
                    image_ = (image[random.randint(0, examples)] * 255).astype(dtype=np.uint8)
                    image_ = Image.fromarray(image_)
                    image_.save(path_, f"{self.file_type}")
                    return path_
                except Exception as e:
                    print(e)
                    raise e
        return


class TinkoffOnceResult(SaveTextOut):
    front_name = 'Первый результат из TinkoffAPI'
    data_type = 'source'

    def execute(self, **kwargs):
        return kwargs.get('tinkoff_first')


class GetText(ClassificationOut):

    front_name = 'Текст'
    data_type = 'test'

    def execute(self, **kwargs):
        result = super().execute(**kwargs)
        output = []
        if isinstance(result, list):
            for result_ in result:
                print(result_)
                while len(result_) == 1:
                    result_ = result_[0]
                output_ = ', '.join([f"{elem[0]} - {elem[1]} %" for elem in result_])
                output.append(output_)
        else:
            output = ', '.join([f"{elem[0]} - {elem[1]} %" for elem in result])
        for string_ in output:
            print(string_)
        return output


class SaveClassificationText(ClassificationOut):

    front_name = 'Файл результатов классификации'
    file_type = 'txt'
    data_type = 'save_classification'

    def execute(self, **kwargs):
        result = super().execute(**kwargs)
        save_path = kwargs.get('save_path')

        output = []
        if isinstance(result, list):
            for result_ in result:
                while len(result_) == 1:
                    result_ = result_[0]
                output_ = ', '.join([f"{elem[0]} - {elem[1]} %" for elem in result_])
                output.append(output_)
        else:
            output = ', '.join([f"{elem[0]} - {elem[1]} %" for elem in result])
        print(output)
        if not os.path.exists(save_path) and not is_terra_file(save_path):
            os.makedirs(save_path, exist_ok=True)

        path_ = save_path if is_terra_file(save_path) else os.path.join(
            save_path, f"predict.{self.file_type}"
        )
        with open(path_, 'a', encoding='utf-8') as text:
            text.write(f'{str([list(result_) for result_ in result[0]])}\n')
        return path_


class ModelOutput(CascadeBlock):
    ImageClassification = [EmptyOut, ImageArrayOut, NativeOut, ClassificationOut, SaveImgOut, GetText]
    ImageSegmentation = [EmptyOut, ImageArrayOut, SegmentationNativeOut, RGBMaskOut,
                         ImageSegmentationOut, SaveImgOut, SaveImageSegmentOut]
    TextClassification = [EmptyOut, TextArrayOut, NativeOut, ClassificationOut, SaveTextOut, SaveClassificationText]
    TextSegmentation = [EmptyOut, TextArrayOut, TextSegmentationNativeOut,
                        TextSegmentationOut, SaveTextOut, SaveTextSegmentationOut]
    TextTransformer = [EmptyOut, TextArrayOut, NativeOut, SaveTextOut]
    DataframeClassification = [EmptyOut, NativeOut, ClassificationOut]
    DataframeRegression = [EmptyOut, NativeOut,]
    Timeseries = [EmptyOut, NativeOut,]
    TimeseriesTrend = [EmptyOut, NativeOut,]
    AudioClassification = [EmptyOut, AudioArrayOut, NativeOut, ClassificationOut, SaveAudioOut, GetText]
    VideoClassification = [EmptyOut, VideoArrayOut, NativeOut, ClassificationOut, SaveVideoOut, GetText]
    YoloV3 = [EmptyOut, ImageArrayOut, YoloNativeOut, SaveImgOut]
    YoloV4 = [EmptyOut, ImageArrayOut, YoloNativeOut, SaveImgOut]
    Tracker = [EmptyOut, NativeOut,]
    Speech2Text = [EmptyOut, NativeOut, SaveAudioOut]
    Text2Speech = [EmptyOut, NativeOut, SaveAudioOut, SaveTextOut]
    TinkoffAPI = [EmptyOut, NativeOut, SaveAudioOut, TinkoffOnceResult]
    ImageGAN = [EmptyOut, ImageArrayOut, SaveImgFromArrayOut]
    ImageCGAN = [EmptyOut, NativeOut,]
    TextToImageGAN = [EmptyOut, ImageArrayOut, SaveImgFromArrayOut]
    ImageToImageGAN = [EmptyOut, NativeOut,]
    PlotBboxes = [ImageArrayOut, SaveImgFromArrayOut, BboxesArrayOut]
    YoloV5 = [EmptyOut, SaveImgFromArrayOut, BboxesArrayOut, InputImageArrayOut]
    DeepSort = [InputImageArrayOut, BboxesArrayOut]
