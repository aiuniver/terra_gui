from terra_ai.data.datasets.extra import LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice, \
    LayerScalerImageChoice, LayerScalerVideoChoice, LayerPrepareMethodChoice
from terra_ai.data.datasets.creation import CreationInputData, CreationOutputData
from terra_ai.data.datasets.extra import LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice,\
    LayerTextModeChoice, LayerAudioModeChoice, LayerVideoModeChoice, LayerScalerAudioChoice

import os
import re
import cv2
import numpy as np
import pandas as pd
import random
import joblib
import pymorphy2
import librosa.feature as librosa_feature
from sklearn.cluster import KMeans
from pydub import AudioSegment
# import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from librosa import load as librosa_load
from pydantic.color import Color
from pydantic import DirectoryPath
from typing import Union
from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import utils


class CreateArray(object):

    @staticmethod
    def instructions_image(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        instructions: dict = {}
        options = put_data.parameters.native()
        del options['augmentation']
        options['put'] = put_data.id

        instructions['parameters'] = options
        instructions['instructions'] = paths_list

        return instructions

    @staticmethod
    def instructions_video(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        instructions: dict = {}
        video: list = []
        cur_step = 0

        for elem in paths_list:
            name, ext = os.path.splitext(os.path.basename(elem))
            if put_data.parameters.video_mode == LayerVideoModeChoice.completely:
                video.append(
                    os.path.join(os.path.dirname(elem), f'{name}_[{cur_step}-{put_data.parameters.max_frames}]{ext}'))
            elif put_data.parameters.video_mode == LayerVideoModeChoice.length_and_step:
                cur_step = 0
                stop_flag = False
                cap = cv2.VideoCapture(elem)
                frame_count = int(cap.get(7))
                while not stop_flag:
                    video.append(os.path.join(os.path.dirname(elem),
                                              f'{name}_[{cur_step}-{cur_step + put_data.parameters.length}]{ext}'))

                    cur_step += put_data.parameters.step
                    if cur_step + put_data.parameters.length > frame_count:
                        stop_flag = True
                        # if put_data.parameters.length < frame_count:
                        #     video.append(os.path.join(os.path.dirname(elem), f'{name}_[{frame_count - put_data.parameters.length}, {frame_count}]{ext}'))
                        #     y_cls.append(csv_y_cls[idx]) if csv_flag else y_cls.append(cur_class)

        options = put_data.parameters.native()
        del options['video_mode']
        del options['length']
        del options['step']
        del options['max_frames']
        options['put'] = put_data.id

        instructions['parameters'] = options
        instructions['instructions'] = video

        return instructions

    @staticmethod
    def instructions_audio(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        instructions: dict = {}
        audio: list = []

        for elem in paths_list:
            name, ext = os.path.splitext(os.path.basename(elem))
            if put_data.parameters.audio_mode == LayerAudioModeChoice.completely:
                audio.append(
                    os.path.join(os.path.dirname(elem), f'{name}_[0.0-{put_data.parameters.max_seconds}]{ext}'))
            elif put_data.parameters.audio_mode == LayerAudioModeChoice.length_and_step:
                cur_step = 0.0
                stop_flag = False
                sample_length = AudioSegment.from_file(elem).duration_seconds
                while not stop_flag:
                    audio.append(os.path.join(os.path.dirname(elem),
                                              f'{name}_[{cur_step}-{put_data.parameters.max_seconds}]{ext}'))
                    cur_step += put_data.parameters.step
                    if cur_step + put_data.parameters.length > sample_length:
                        stop_flag = True

        options = put_data.parameters.native()
        for elem in ['audio_mode', 'file_info', 'length', 'step', 'max_seconds']:
            if elem in options.keys():
                del options[elem]
        options['put'] = put_data.id

        instructions['parameters'] = options
        instructions['instructions'] = audio

        return instructions

    @staticmethod
    def instructions_text(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        def read_text(file_path, lower, del_symbols, split, open_symbol=None, close_symbol=None) -> str:

            with open(file_path, 'r', encoding='utf-8') as txt:
                text = txt.read()

            if open_symbol:
                text = re.sub(open_symbol, f" {open_symbol}", text)
                text = re.sub(close_symbol, f"{close_symbol} ", text)

            text = ' '.join(text_to_word_sequence(text, **{'lower': lower, 'filters': del_symbols, 'split': split}))

            return text

        def apply_pymorphy(text, morphy) -> str:

            words_list = text.split(' ')
            words_list = [morphy.parse(w)[0].normal_form for w in words_list]

            return ' '.join(words_list)

        instructions: dict = {}
        txt_list: dict = {}
        text: list = []
        lower: bool = True
        open_tags, close_tags = None, None
        open_symbol, close_symbol = None, None
        if put_data.parameters.open_tags:
            open_tags, close_tags = put_data.parameters.open_tags.split(' '), put_data.parameters.close_tags.split(' ')
        if open_tags:
            open_symbol = open_tags[0][0]
            close_symbol = close_tags[0][-1]

        for idx, path in enumerate(paths_list):
            txt_list[path] = read_text(file_path=path, lower=lower, del_symbols=put_data.parameters.filters, split=' ',
                                       open_symbol=open_symbol, close_symbol=close_symbol)

        if open_symbol:
            for key in txt_list.keys():
                words = []
                for word in txt_list[key].split(' '):
                    if word not in open_tags + close_tags:
                        words.append(word)
                txt_list[key] = ' '.join(words)

        if put_data.parameters.pymorphy:
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_list.items():
                txt_list[key] = apply_pymorphy(value, pymorphy)

        for idx, (key, value) in enumerate(sorted(txt_list.items())):
            if put_data.parameters.text_mode == LayerTextModeChoice.completely:
                text.append(' '.join(value.split(' ')[:put_data.parameters.max_words]))
            elif put_data.parameters.text_mode == LayerTextModeChoice.length_and_step:
                max_length = len(value.split(' '))
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text.append(' '.join(value.split(' ')[cur_step: cur_step + put_data.parameters.length]))
                    cur_step += put_data.parameters.step
                    if cur_step + put_data.parameters.length > max_length:
                        stop_flag = True

        options = put_data.parameters.native()
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else options[
            'max_words']

        instructions['parameters'] = {'prepare_method': put_data.parameters.prepare_method,
                                      'put': put_data.id,
                                      'length': length,
                                      'max_words_count': put_data.parameters.max_words_count,
                                      'word_to_vec_size': put_data.parameters.word_to_vec_size,
                                      'filters': put_data.parameters.filters
                                      }
        instructions['instructions'] = text

        return instructions

    @staticmethod
    def instructions_classification(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        instructions: dict = {}
        if put_data.parameters.sources_paths[0].is_file() and put_data.parameters.sources_paths[0].suffix == ".csv":
            file_name = put_data.parameters.sources_paths[0]
            data = pd.read_csv(file_name, usecols=put_data.parameters.cols_names)
            column = data[put_data.parameters.cols_names[0]].to_list()
            classes_names = []
            for elem in column:
                if elem not in classes_names:
                    classes_names.append(elem)
            num_classes = len(classes_names)
        else:
            classes_names = sorted([os.path.basename(elem) for elem in put_data.parameters.sources_paths])
            num_classes = len(classes_names)

        instructions["parameters"] = {"one_hot_encoding": put_data.parameters.one_hot_encoding,
                                      "classes_names": classes_names,
                                      "num_classes": num_classes,
                                      'put': put_data.id
                                      }
        instructions['instructions'] = paths_list

        return instructions

    @staticmethod
    def instructions_regression(number_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        options = put_data.parameters.native()
        options["put"] = put_data.id

        instructions: dict = {'parameters': options,
                              'instructions': number_list}

        return instructions

    @staticmethod
    def instructions_segmentation(paths_list: list, put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        instructions: dict = {}
        instructions['parameters'] = {'mask_range': put_data.parameters.mask_range,
                                      'num_classes': len(put_data.parameters.classes_names),
                                      'height': put_data.parameters.height,
                                      'width': put_data.parameters.width,
                                      'classes_colors': [Color(color).as_rgb_tuple() for color in
                                                         put_data.parameters.classes_colors],
                                      'put': put_data.id
                                      }
        instructions['instructions'] = paths_list

        return instructions

    @staticmethod
    def instructions_text_segmentation(paths_list: list,
                                       put_data: Union[CreationInputData, CreationOutputData]) -> dict:

        """

        Args:
            **put_data:
                open_tags: str
                    Открывающие теги.
                close_tags: str
                    Закрывающие теги.

        Returns:

        """

        def read_text(file_path, lower, del_symbols, split, open_symbol=None, close_symbol=None) -> str:

            with open(file_path, 'r', encoding='utf-8') as txt:
                text = txt.read()

            if open_symbol:
                text = re.sub(open_symbol, f" {open_symbol}", text)
                text = re.sub(close_symbol, f"{close_symbol} ", text)

            text = ' '.join(text_to_word_sequence(text, **{'lower': lower, 'filters': del_symbols, 'split': split}))

            return text

        def get_samples(doc_text: str, op_tags, cl_tags):

            words = []
            indexes = []
            idx = []
            for word in doc_text.split(' '):
                try:
                    if word in op_tags:
                        idx.append(op_tags[op_tags.index(word)])
                    elif word in cl_tags:
                        idx.remove(op_tags[cl_tags.index(word)])
                    else:
                        indexes.append(idx.copy())
                except ValueError:
                    pass
                    # print(word)

            return indexes

        options = put_data.parameters.native()
        instructions: dict = {}
        text_segm: list = []
        text_segm_data: list = []
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        for elem in paths_list:
            text = read_text(elem, True, options['filters'], ' ', open_symbol, close_symbol)
            text_segm.append(get_samples(text, open_tags, close_tags))

        for text in text_segm:
            if options['text_mode'] == LayerTextModeChoice.completely:
                text_segm_data.append(text[0:options['max_words']])
            elif options['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(text)
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text_segm_data.append(text[cur_step:cur_step + options['length']])
                    cur_step += options['step']
                    if cur_step + options['length'] > max_length:
                        stop_flag = True

        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else options[
            'max_words']

        instructions['parameters'] = {'num_classes': len(open_tags),
                                      'classes_names': open_tags,
                                      'put': put_data.id,
                                      'length': length
                                      }
        instructions['instructions'] = text_segm_data

        return instructions

    @staticmethod
    def create_image(image_path: str, **options: dict) -> np.ndarray:

        img = load_img(image_path, target_size=(options['height'], options['width']))
        array = img_to_array(img, dtype=np.uint8)
        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options['object_scaler']:
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def create_video(video_path: str, **options: dict) -> np.ndarray:

        def resize_frame(one_frame, original_shape, target_shape, frame_mode):

            resized = None

            if frame_mode == LayerVideoFrameModeChoice.stretch:
                resized = resize_layer(one_frame[None, ...])
                resized = resized.numpy().squeeze().astype('uint8')
            elif frame_mode == LayerVideoFrameModeChoice.keep_proportions:
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

            if fill_mode == LayerVideoFillModeChoice.black_frames:
                frames = np.zeros((frames_to_add, *shape, 3), dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.average_value:
                mean = np.mean(video_array, axis=0, dtype='uint16')
                frames = np.full((frames_to_add, *mean.shape), mean, dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.last_frames:
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
        slicing = [int(x) for x in video_path[video_path.index('[') + 1:video_path.index(']')].split('-')]
        frames_count = slicing[1] - slicing[0]
        resize_layer = Resizing(*shape)

        cap = cv2.VideoCapture(video_path)
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

        if options['scaler'] != LayerScalerVideoChoice.no_scaler and options['object_scaler']:
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def create_audio(audio_path: str, **options: dict) -> np.ndarray:

        array = []
        parameter = options['parameter']
        sample_rate = options['sample_rate']
        slicing = [float(x) for x in audio_path[audio_path.index('[') + 1:audio_path.index(']')].split('-')]
        y, sr = librosa_load(path=audio_path, sr=options.get('sample_rate'),
                             offset=slicing[0], duration=slicing[1] - slicing[0], res_type='kaiser_best')
        if sample_rate > len(y):
            zeros = np.zeros((sample_rate - len(y),))
            y = np.concatenate((y, zeros))

        if parameter in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
            array = getattr(librosa_feature, parameter)(y=y, sr=sr)
        elif parameter == 'rms':
            array = getattr(librosa_feature, parameter)(y=y)[0]
        elif parameter == 'zero_crossing_rate':
            array = getattr(librosa_feature, parameter)(y=y)
        elif parameter == 'audio_signal':
            array = y

        array = np.array(array)
        if array.dtype == 'float64':
            array = array.astype('float32')

        if options['scaler'] != LayerScalerAudioChoice.no_scaler and options['object_scaler']:
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def create_text(text: str, **options: dict) -> np.ndarray:

        array = []
        text = text.split(' ')

        if options['prepare_method'] == LayerPrepareMethodChoice.embedding:
            array = options['object_tokenizer'].texts_to_sequences([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.bag_of_words:
            array = options['object_tokenizer'].texts_to_matrix([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
            for word in text:
                array.append(options['object_word2vec'][word])

        if len(array) < options['length']:
            if options['prepare_method'] in [LayerPrepareMethodChoice.embedding, LayerPrepareMethodChoice.bag_of_words]:
                words_to_add = [0 for _ in range((options['length']) - len(array))]
            elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                words_to_add = [[0 for _ in range(options['word_to_vec_size'])] for _ in
                                range((options['length']) - len(array))]
            array += words_to_add

        array = np.array(array)

        return array

    @staticmethod
    def create_classification(class_name: str, **options) -> np.ndarray:

        index = options['classes_names'].index(os.path.basename(class_name))
        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        index = np.array(index)

        return index

    @staticmethod
    def create_regression(index: int, **options) -> np.ndarray:

        array = np.array(index)

        return array

    @staticmethod
    def create_segmentation(image_path: str, **options: dict) -> np.ndarray:

        def cluster_to_ohe(mask_image):

            mask_image = mask_image.reshape(-1, 3)
            km = KMeans(n_clusters=options['num_classes'])
            km.fit(mask_image)
            labels = km.labels_
            cl_cent = km.cluster_centers_.astype('uint8')[:max(labels) + 1]
            cl_mask = utils.to_categorical(labels, max(labels) + 1, dtype='uint8')
            cl_mask = cl_mask.reshape(options['height'], options['width'], cl_mask.shape[-1])
            mask_ohe = np.zeros((options['height'], options['width']))
            for k, color in enumerate(options['classes_colors']):
                rgb = Color(color).as_rgb_tuple()
                mask = np.zeros((options['height'], options['width']))

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

        img = load_img(path=image_path, target_size=(options['height'], options['width']))
        array = img_to_array(img, dtype=np.uint8)
        array = cluster_to_ohe(array)

        return array

    @staticmethod
    def create_text_segmentation(text: list, **options) -> np.ndarray:

        array = []
        if len(text) < options['length']:
            text += [list() for _ in range(options['length'] - len(text))]
        for elem in text:
            tags = [0 for _ in range(options['num_classes'])]
            if elem:
                for cls_name in elem:
                    tags[options['classes_names'].index(cls_name)] = 1
            array.append(tags)
        array = np.array(array, dtype='uint8')

        return array
