import colorsys
import copy
import math
import string
from typing import Optional

import matplotlib
from PIL import Image, UnidentifiedImageError, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from pandas import DataFrame
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice, ArchitectureChoice
from terra_ai.datasets.utils import get_yolo_anchors
from terra_ai.data.datasets.dataset import DatasetOutputsData, DatasetData
from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerScalerVideoChoice, LayerPrepareMethodChoice, \
    LayerOutputTypeChoice, DatasetGroupChoice, LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.datasets.extra import LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice, \
    LayerTextModeChoice, LayerAudioModeChoice, LayerVideoModeChoice, LayerScalerAudioChoice

import os
import re
import cv2
import numpy as np
import pandas as pd
import shutil
import pymorphy2
import random
import librosa.feature as librosa_feature
from ast import literal_eval
from sklearn.cluster import KMeans
from pydub import AudioSegment
from librosa import load as librosa_load
from pydantic.color import Color
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import utils
from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum
import moviepy.editor as moviepy_editor

from terra_ai.settings import DEPLOY_PRESET_PERCENT, CALLBACK_CLASSIFICATION_TREASHOLD_VALUE, \
    CALLBACK_REGRESSION_TREASHOLD_VALUE


def print_error(class_name: str, method_name: str, message: Exception):
    return print(f'\n_________________________________________________\n'
                 f'Error in class {class_name} method {method_name}: {message}'
                 f'\n_________________________________________________\n')


class CreateArray(object):

    @staticmethod
    def instructions_image(paths_list: list, **options: dict) -> dict:

        p_list = []
        for elem in paths_list:
            try:
                load_img(elem).verify()
                p_list.append(elem)
            except (UnidentifiedImageError, IOError):
                pass

        instructions = {'instructions': p_list,
                        'parameters': options
                        }

        return instructions

    @staticmethod
    def instructions_video(paths_list: list, **options) -> dict:

        video: list = []
        cur_step = 0

        for elem in paths_list:
            cap = cv2.VideoCapture(elem)
            if cap.isOpened():
                if options['video_mode'] == LayerVideoModeChoice.completely:
                    video.append(';'.join([elem, f'[{cur_step}-{options["max_frames"]}]']))
                elif options['video_mode'] == LayerVideoModeChoice.length_and_step:
                    cur_step = 0
                    stop_flag = False
                    cap = cv2.VideoCapture(elem)
                    frame_count = int(cap.get(7))
                    while not stop_flag:
                        video.append(';'.join([elem, f'[{cur_step}-{cur_step + options["length"]}]']))
                        cur_step += options['step']
                        if cur_step + options['length'] > frame_count:
                            stop_flag = True
                            if options['length'] < frame_count:
                                video.append(
                                    ';'.join([elem, f'[{frame_count - options["length"]}-{frame_count}]']))

        instructions = {'instructions': video,
                        'parameters': options
                        }

        return instructions

    @staticmethod
    def instructions_audio(paths_list: list, **options) -> dict:

        audio: list = []

        for elem in paths_list:
            try:
                librosa_load(elem, duration=0.002, res_type='scipy')  # Проверка файла на аудио-формат.
                if options['audio_mode'] == LayerAudioModeChoice.completely:
                    audio.append(';'.join([elem, f'[0.0-{options["max_seconds"]}]']))
                elif options['audio_mode'] == LayerAudioModeChoice.length_and_step:
                    cur_step = 0.0
                    stop_flag = False
                    sample_length = AudioSegment.from_file(elem).duration_seconds
                    while not stop_flag:
                        audio.append(';'.join([elem, f'[{cur_step}-{round(cur_step + options["length"], 1)}]']))
                        cur_step += options['step']
                        cur_step = round(cur_step, 1)
                        if cur_step + options['length'] > sample_length:
                            stop_flag = True
            except:
                pass

        instructions = {'instructions': audio,
                        'parameters': {**options,
                                       'duration': options['max_seconds'] if options['audio_mode'] == 'completely' else
                                       options['length']
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_text(text_list: list, **options) -> dict:

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

        txt_dict: dict = {}
        text: dict = {}
        lower: bool = True
        open_tags, close_tags = None, None
        open_symbol, close_symbol = None, None
        if options.get('open_tags'):
            open_tags, close_tags = options['open_tags'].split(' '), options['close_tags'].split(' ')
            # if open_tags:
            open_symbol = open_tags[0][0]
            close_symbol = close_tags[0][-1]
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['max_words']

        for idx, text_row in enumerate(text_list):
            if os.path.isfile(str(text_row)):
                text_file = read_text(file_path=text_row, lower=lower, del_symbols=options['filters'], split=' ',
                                      open_symbol=open_symbol, close_symbol=close_symbol)
                if text_file:
                    txt_dict[text_row] = text_file
            else:
                if not text_row:
                    txt_dict[idx] = "nan"
                elif not isinstance(text_row, str):
                    txt_dict[idx] = str(text_row)
                else:
                    txt_dict[idx] = text_row

        if open_symbol:
            for key in txt_dict.keys():
                words = []
                for word in txt_dict[key].split(' '):
                    if word not in open_tags + close_tags:
                        words.append(word)
                txt_dict[key] = ' '.join(words)

        if options['pymorphy']:
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_dict.items():
                txt_dict[key] = apply_pymorphy(value, pymorphy)

        for key, value in sorted(txt_dict.items()):
            if options['text_mode'] == LayerTextModeChoice.completely:
                text[';'.join([str(key), f'[0-{options["max_words"]}]'])] = ' '.join(
                    value.split(' ')[:options['max_words']])
            elif options['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(value.split(' '))
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text[';'.join([str(key), f'[{cur_step}-{cur_step + length}]'])] = ' '.join(
                        value.split(' ')[cur_step: cur_step + length])
                    cur_step += options['step']
                    if cur_step + options['length'] > max_length:
                        stop_flag = True

        instructions = {'instructions': text,
                        'parameters': {**options,
                                       'length': length,
                                       'word_to_vec_size': options.get('word_to_vec_size'),
                                       },
                        }

        return instructions

    @staticmethod
    def instructions_scaler(number_list: list, **options: dict) -> dict:

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def instructions_classification(paths_list: list, **options) -> dict:

        length = options['length'] if 'length' in options.keys() else None
        depth = options['depth'] if 'depth' in options.keys() else None
        step = options['step'] if 'step' in options.keys() else None

        type_processing = options['type_processing']

        if 'sources_paths' in options.keys():
            classes_names = sorted([os.path.basename(elem) for elem in options['sources_paths']])
        else:
            if type_processing == "categorical":
                classes_names = list(dict.fromkeys(paths_list))
            else:
                if len(options["ranges"].split(" ")) == 1:
                    border = max(paths_list) / int(options["ranges"])
                    classes_names = np.linspace(border, max(paths_list), int(options["ranges"])).tolist()
                else:
                    classes_names = options["ranges"].split(" ")

        instructions = {'instructions': paths_list,
                        'parameters': {**options,
                                       "classes_names": classes_names,
                                       "num_classes": len(classes_names),
                                       'type_processing': type_processing,
                                       'length': length,
                                       'step': step,
                                       'depth': depth
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_regression(number_list: list, **options: dict) -> dict:

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def instructions_segmentation(paths_list: list, **options: dict) -> dict:

        p_list = []
        for elem in paths_list:
            try:
                load_img(elem).verify()
                p_list.append(elem)
            except (UnidentifiedImageError, IOError):
                pass

        instructions = {'instructions': p_list,
                        'parameters': {**options,
                                       'classes_colors': [Color(color).as_rgb_tuple() for color in
                                                          options['classes_colors']],
                                       'num_classes': len(options['classes_names'])}
                        }

        return instructions

    @staticmethod
    def instructions_text_segmentation(paths_list: list, **options) -> dict:

        """

        Args:
            paths_list: list
                Пути к файлам.
            **options:
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

            return indexes

        text_list: dict = {}
        text_segm_data: dict = {}
        open_tags: list = options['open_tags'].split(' ')
        close_tags: list = options['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        length = options['length'] if options['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['max_words']

        for path in paths_list:
            text_file = read_text(file_path=path, lower=True, del_symbols=options['filters'], split=' ',
                                  open_symbol=open_symbol, close_symbol=close_symbol)
            if text_file:
                text_list[path] = get_samples(text_file, open_tags, close_tags)

        for key, value in sorted(text_list.items()):
            if options['text_mode'] == LayerTextModeChoice.completely:
                text_segm_data[';'.join([key, f'[0-{options["max_words"]}]'])] = \
                    value[:options['max_words']]
            elif options['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(value)
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text_segm_data[';'.join([key, f'[{cur_step}-{cur_step + length}]'])] = value[
                                                                                           cur_step:cur_step + length]
                    cur_step += options['step']
                    if cur_step + length > max_length:
                        stop_flag = True

        instructions = {'instructions': text_segm_data,
                        'parameters': {**options,
                                       'num_classes': len(open_tags),
                                       'classes_names': open_tags,
                                       'length': length
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_timeseries(number_list, **options: dict) -> dict:

        instructions = {'instructions': number_list,
                        'parameters': options}
        if options['trend']:
            classes = []
            trend_dict = {0: "Не изменился",
                          1: "Вверх",
                          2: "Вниз"}
            tmp = []
            depth = 1
            step = options['step']
            length = options['length']
            trend_lim = options["trend_limit"]
            trend_limit = float(trend_lim[: trend_lim.find("%")]) if "%" in trend_lim else float(trend_lim)
            for i in range(0, len(number_list) - length - depth, step):
                first_value = number_list[i]
                second_value = number_list[i + length]
                if "%" in trend_lim:
                    if abs((second_value - first_value) / first_value) * 100 <= trend_limit:
                        tmp.append(0)
                    elif second_value > first_value:
                        tmp.append(1)
                    else:
                        tmp.append(2)
                else:
                    if abs(second_value - first_value) <= trend_limit:
                        tmp.append(0)
                    elif second_value > first_value:
                        tmp.append(1)
                    else:
                        tmp.append(2)

            for i in set(tmp):
                classes.append(trend_dict[i])
            instructions['parameters']['classes_names'] = classes
            instructions['parameters']['num_classes'] = len(classes)
        return instructions

    @staticmethod
    def instructions_object_detection(paths_list: list, **options: dict) -> dict:

        coordinates_list = []
        for path in paths_list:
            with open(path, 'r') as coordinates:
                coordinate = coordinates.read()

            coordinates_list.append(' '.join([coord for coord in coordinate.split('\n') if coord]))

        instructions = {'instructions': coordinates_list,
                        'parameters': options
                        }

        return instructions

    @staticmethod
    def cut_image(paths_list: list, dataset_folder=None, **options: dict):

        for elem in paths_list:
            os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
            shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                               os.path.basename(elem)))

        paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                      for elem in paths_list]

        instructions = {'instructions': paths_list,
                        'parameters': {'height': options['height'],
                                       'width': options['width'],
                                       'net': options['net'],
                                       # 'object_detection': options['object_detection'],
                                       'scaler': options['scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'min_scaler': options['min_scaler'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'augmentation': options.get('augmentation')
                                       }
                        }

        return instructions

    @staticmethod
    def cut_video(paths_list: list, dataset_folder=None, **options):

        def add_frames(video_array, fill_mode, frames_to_add, total_frames):

            frames: np.ndarray = np.array([])

            if fill_mode == LayerVideoFillModeChoice.last_frames:
                frames = np.full((frames_to_add, *video_array[-1].shape), video_array[-1], dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.average_value:
                mean = np.mean(video_array, axis=0, dtype='uint16')
                frames = np.full((frames_to_add, *mean.shape), mean, dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.loop:
                current_frames = (total_frames - frames_to_add)
                if current_frames > frames_to_add:
                    frames = np.flip(video_array[-frames_to_add:], axis=0)
                elif current_frames <= frames_to_add:
                    for i in range(frames_to_add // current_frames):
                        frames = np.flip(video_array[-current_frames:], axis=0)
                        video_array = np.concatenate((video_array, frames), axis=0)
                    if frames_to_add + current_frames != video_array.shape[0]:
                        frames = np.flip(video_array[-(frames_to_add + current_frames - video_array.shape[0]):], axis=0)
                        video_array = np.concatenate((video_array, frames), axis=0)
                    frames = video_array[current_frames:]

            return frames

        instructions_paths = []

        for elem in paths_list:
            tmp_array = []
            os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))),
                        exist_ok=True)
            path, slicing = elem.split(';')
            slicing = [int(x) for x in slicing[1:-1].split('-')]
            name, ext = os.path.splitext(os.path.basename(path))
            cap = cv2.VideoCapture(path)
            cap.set(1, slicing[0])
            orig_shape = (int(cap.get(3)), int(cap.get(4)))
            frames_number = 0
            save_path = os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                     f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
            instructions_paths.append(save_path)
            output_movie = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), orig_shape)
            stop_flag = False
            while not stop_flag:
                ret, frame = cap.read()
                if not ret or frames_number == slicing[1] - slicing[0]:  # frames_number > frames_count or
                    stop_flag = True
                else:
                    output_movie.write(frame)
                    tmp_array.append(frame)
                    frames_number += 1
            if options['video_mode'] == 'completely' and options['max_frames'] > frames_number or \
                    options['video_mode'] == 'length_and_step' and options['length'] > frames_number:
                fr_to_add, tot_frames = 0, 0
                if options['video_mode'] == 'completely':
                    fr_to_add = options['max_frames'] - frames_number
                    tot_frames = options['max_frames']
                elif options['video_mode'] == 'length_and_step':
                    fr_to_add = options['length'] - frames_number
                    tot_frames = options['length']
                frames_to_add = add_frames(video_array=np.array(tmp_array),
                                           fill_mode=options['fill_mode'],
                                           frames_to_add=fr_to_add,
                                           total_frames=tot_frames)
                for arr in frames_to_add:
                    output_movie.write(arr)

            output_movie.release()

        instructions = {'instructions': instructions_paths,
                        'parameters': {'height': options['height'],
                                       'width': options['width'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'min_scaler': options['min_scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'scaler': options['scaler'],
                                       'frame_mode': options['frame_mode'],
                                       'fill_mode': options['fill_mode'],
                                       'video_mode': options['video_mode'],
                                       'length': options['length'],
                                       'max_frames': options['max_frames']}
                        }

        return instructions

    @staticmethod
    def cut_audio(paths_list: list, dataset_folder=None, **options: dict):

        instructions_paths = []
        for elem in paths_list:
            path, slicing = elem.split(';')
            name, ext = os.path.splitext(os.path.basename(path))
            slicing = [float(x) for x in slicing[1:-1].split('-')]
            duration = round(slicing[1] - slicing[0], 1)
            os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(path))),
                        exist_ok=True)
            audio = AudioSegment.from_file(path, start_second=slicing[0], duration=duration)

            if round(duration - audio.duration_seconds, 3) != 0:
                while not round(duration - audio.duration_seconds, 3):
                    if options['fill_mode'] == 'last_millisecond':
                        audio = audio.append(audio[-2], crossfade=0)
                    elif options['fill_mode'] == 'loop':
                        duration_to_add = round(duration - audio.duration_seconds, 3)
                        if audio.duration_seconds < duration_to_add:
                            audio = audio.append(audio[0:audio.duration_seconds * 1000], crossfade=0)
                        else:
                            audio = audio.append(audio[0:duration_to_add * 1000], crossfade=0)

            save_path = os.path.join(dataset_folder, os.path.basename(os.path.dirname(path)),
                                     f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
            audio.export(save_path, format=ext[1:])
            instructions_paths.append(save_path)

        instructions = {'instructions': instructions_paths,
                        'parameters': {'duration': options['duration'],
                                       'sample_rate': options['sample_rate'],
                                       'resample': options['resample'],
                                       'parameter': options['parameter'],
                                       'cols_names': options['cols_names'],
                                       'scaler': options['scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'min_scaler': options['min_scaler'],
                                       'put': options['put']}}

        return instructions

    @staticmethod
    def cut_text(paths_list: dict, dataset_folder=None, **options: dict):

        text_list = []
        for elem in sorted(paths_list.keys()):
            text_list.append(paths_list[elem])

        instructions = {'instructions': text_list,
                        'parameters': {'prepare_method': options['prepare_method'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'length': options['length'],
                                       'max_words_count': options['max_words_count'],
                                       'word_to_vec_size': options['word_to_vec_size'],
                                       'filters': options['filters']
                                       }
                        }

        return instructions

    @staticmethod
    def cut_scaler(number_list: list, dataset_folder=None, **options: dict):

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_classification(paths_list: list, dataset_folder=None, **options: dict):

        instructions = {'instructions': paths_list,
                        'parameters': {"classes_names": options['classes_names'],
                                       "num_classes": options['num_classes'],
                                       'cols_names': options['cols_names'],
                                       'put': options['put'],
                                       'type_processing': options['type_processing'],
                                       'length': options['length'],
                                       'step': options['step'],
                                       'depth': options['depth']
                                       }
                        }

        return instructions

    @staticmethod
    def cut_regression(number_list: list, dataset_folder=None, **options: dict):

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_segmentation(paths_list: list, dataset_folder=None, **options: dict):

        for elem in paths_list:
            os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
            shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                               os.path.basename(elem)))

        paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                      for elem in paths_list]

        instructions = {'instructions': paths_list,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': options['num_classes'],
                                       'height': options['height'],
                                       'width': options['width'],
                                       'classes_colors': options['classes_colors'],
                                       'classes_names': options['classes_names'],
                                       'cols_names': options['cols_names'],
                                       'put': options['put']
                                       }
                        }

        return instructions

    @staticmethod
    def cut_text_segmentation(paths_list: dict, dataset_folder=None, **options: dict):

        text_list = []
        for elem in sorted(paths_list.keys()):
            text_list.append(paths_list[elem])

        instructions = {'instructions': text_list,
                        'parameters': {'open_tags': options['open_tags'],
                                       'close_tags': options['close_tags'],
                                       'put': options['put'],
                                       'num_classes': options['num_classes'],
                                       'classes_names': options['classes_names'],
                                       'length': options['length']
                                       }
                        }

        return instructions

    @staticmethod
    def cut_timeseries(paths_list: dict, dataset_folder=None, **options: dict):

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_object_detection(bounding_boxes: list, dataset_folder=None, **options: dict) -> dict:

        # for elem in paths_list:
        #     os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
        #     shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
        #                                        os.path.basename(elem)))
        #
        # paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
        #               for elem in paths_list]

        instructions = {'instructions': bounding_boxes,
                        'parameters': {'yolo': options['yolo'],
                                       'num_classes': options['num_classes'],
                                       'classes_names': options['classes_names'],
                                       'put': options['put']}
                        }

        return instructions

    @staticmethod
    def create_image(image_path: str, **options) -> dict:

        img = load_img(image_path)
        array = np.array(img)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_video(video_path: str, **options) -> dict:

        array = []
        slicing = [int(x) for x in video_path[video_path.index('[') + 1:video_path.index(']')].split('-')]
        frames_count = slicing[1] - slicing[0]
        cap = cv2.VideoCapture(video_path)
        try:
            for _ in range(frames_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame[:, :, [2, 1, 0]]
                array.append(frame)
        finally:
            cap.release()

        array = np.array(array)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_audio(audio_path: str, **options) -> dict:

        array = []
        parameter = options['parameter']
        sample_rate = options['sample_rate']
        y, sr = librosa_load(path=audio_path, sr=options.get('sample_rate'), res_type=options.get('resample'))
        if round(sample_rate * options['duration'], 0) > y.shape[0]:
            zeros = np.zeros((int(round(sample_rate * options['duration'], 0)) - y.shape[0],))
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
        if len(array.shape) == 2:
            array = array.transpose()
        if array.dtype == 'float64':
            array = array.astype('float32')

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_text(text: str, **options) -> dict:

        instructions = {'instructions': text,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_scaler(index: int, **options) -> dict:

        instructions = {'instructions': np.array(index),
                        'parameters': options}

        return instructions

    @staticmethod
    def create_classification(class_name, **options) -> dict:

        class_name = class_name.to_list() if isinstance(class_name, pd.Series) else class_name
        class_name = class_name if isinstance(class_name, list) else [class_name]
        if options['type_processing'] == 'categorical':
            if len(class_name) == 1:
                index = [options['classes_names'].index(class_name[0])]
            else:
                index = []
                for i in range(len(class_name)):
                    index.append(options['classes_names'].index(class_name[i]))
        else:
            index = []
            for i in range(len(class_name)):
                for j, cl_name in enumerate(options['classes_names']):
                    if class_name[i] <= float(cl_name):
                        index.append(j)
                        break
        if len(class_name) == 1:
            index = utils.to_categorical(index[0], num_classes=options['num_classes'], dtype='uint8')
        else:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')

        index = np.array(index)

        instructions = {'instructions': index,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_regression(index: int, **options) -> dict:

        instructions = {'instructions': np.array([index]),
                        'parameters': options}

        return instructions

    @staticmethod
    def create_segmentation(image_path: str, **options) -> dict:

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
        array = np.array(img)
        array = cluster_to_ohe(array)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_text_segmentation(text, **options) -> dict:

        if not isinstance(text, list):
            text = literal_eval(text)
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

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_timeseries(row, **options) -> dict:

        if options["trend"]:
            trend_dict = {0: "Не изменился",
                          1: "Вверх",
                          2: "Вниз"}
            first_value = row[0]
            second_value = row[1]

            trend_limit = options["trend_limit"]
            if "%" in trend_limit:
                trend_limit = float(trend_limit[: trend_limit.find("%")])
                if abs((second_value - first_value) / first_value) * 100 <= trend_limit:
                    array = 0
                elif second_value > first_value:
                    array = 1
                else:
                    array = 2
            else:
                trend_limit = float(trend_limit)
                if abs(second_value - first_value) <= trend_limit:
                    array = 0
                elif second_value > first_value:
                    array = 1
                else:
                    array = 2
            idx = options['classes_names'].index(trend_dict[array])
            array = utils.to_categorical(idx, num_classes=options['num_classes'], dtype='uint8')
        else:
            array = row

        instructions = {'instructions': np.array(array),
                        'parameters': options}

        return instructions

    @staticmethod
    def create_object_detection(coords: str, **options):

        """
        Args:
            coords: str
                Координаты bounding box.
            **options:
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

            boxes1 = tf_concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
            boxes2 = tf_concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

            left_up = tf_maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = tf_minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = tf_maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area

            return 1.0 * inter_area / union_area

        # height: int = options['height']
        # width: int = options['width']
        x_scale = options['orig_x'] / 416
        y_scale = options['orig_y'] / 416

        real_boxes = []
        for coord in coords.split(' '):
            tmp = []
            for i, num in enumerate(coord.split(',')):
                if i in [0, 2]:
                    tmp_value = int(literal_eval(num) / x_scale) - 1
                    scale_value = options['orig_x'] if tmp_value > options['orig_x'] else tmp_value
                    tmp.append(scale_value)
                elif i in [1, 3]:
                    tmp_value = int(literal_eval(num) / y_scale) - 1
                    scale_value = options['orig_y'] if tmp_value > options['orig_y'] else tmp_value
                    tmp.append(scale_value)
                else:
                    tmp.append(literal_eval(num))
            real_boxes.append(tmp)

        num_classes: int = options['num_classes']
        zero_boxes_flag: bool = False
        strides = np.array([8, 16, 32])
        output_levels = len(strides)
        train_input_sizes = 416
        anchor_per_scale = 3

        yolo_anchors = get_yolo_anchors(options['yolo'])

        # if options['yolo'] == 'v3':
        #     yolo_anchors = [[[10, 13], [16, 30], [33, 23]],
        #                     [[30, 61], [62, 45], [59, 119]],
        #                     [[116, 90], [156, 198], [373, 326]]]
        # elif options['yolo'] == 'v4':
        #     yolo_anchors = [[[12, 16], [19, 36], [40, 28]],
        #                     [[36, 75], [76, 55], [72, 146]],
        #                     [[142, 110], [192, 243], [459, 401]]]

        anchors = (np.array(yolo_anchors).T / strides).T
        max_bbox_per_scale = 100
        train_input_size = random.choice([train_input_sizes])
        train_output_sizes = train_input_size // strides

        label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale,
                           5 + num_classes)) for i in range(output_levels)]
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

            bbox_xywh = np.concatenate([(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                                        bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(output_levels):  # range(3):
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 0.0 if zero_boxes_flag else 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_one_hot

                    bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchor_per_scale)
                best_anchor = int(best_anchor_ind % anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 0.0 if zero_boxes_flag else 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        instructions = {'instructions': [np.array(label_sbbox, dtype='float32'), np.array(label_mbbox, dtype='float32'),
                                         np.array(label_lbbox, dtype='float32'), np.array(sbboxes, dtype='float32'),
                                         np.array(mbboxes, dtype='float32'), np.array(lbboxes, dtype='float32')],
                        'parameters': options}

        return instructions

    @staticmethod
    def create_raw(item, **options) -> dict:
        if isinstance(item, str):
            try:
                item = literal_eval(item)
            except:
                pass
            item = np.array([item])
        elif isinstance(item, list):
            item = np.array(item)
        elif isinstance(item, pd.Series):
            item = item.values
        else:
            item = np.array([item])

        instructions = {'instructions': item,
                        'parameters': options}

        return instructions

    @staticmethod
    def preprocess_image(array: np.ndarray, **options) -> np.ndarray:

        array = cv2.resize(array, (options['width'], options['height']))
        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            if options['scaler'] == 'min_max_scaler':
                orig_shape = array.shape
                array = options['preprocess'].transform(array.reshape(-1, 1))
                array = array.reshape(orig_shape).astype('float32')
            elif options['scaler'] == 'terra_image_scaler':
                array = options['preprocess'].transform(array)

        return array

    @staticmethod
    def preprocess_video(array: np.ndarray, **options) -> np.ndarray:

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

        orig_shape = array.shape[1:]
        trgt_shape = (options['height'], options['width'])
        resize_layer = Resizing(*trgt_shape)
        resized_array = []
        for i in range(len(array)):
            if array[i].shape[1:-1] != trgt_shape:
                resized_array.append(resize_frame(one_frame=array[i],
                                                  original_shape=orig_shape,
                                                  target_shape=trgt_shape,
                                                  frame_mode=options['frame_mode']))
        array = np.array(resized_array)

        if options['scaler'] != LayerScalerVideoChoice.no_scaler and options.get('preprocess'):
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def preprocess_audio(array: np.ndarray, **options) -> np.ndarray:

        if options['scaler'] != LayerScalerAudioChoice.no_scaler and options.get('preprocess'):
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        return array

    @staticmethod
    def preprocess_text(text: str, **options) -> np.ndarray:

        array = []
        text = text.split(' ')
        words_to_add = []

        if options['prepare_method'] == LayerPrepareMethodChoice.embedding:
            array = options['preprocess'].texts_to_sequences([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.bag_of_words:
            array = options['preprocess'].texts_to_matrix([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
            for word in text:
                try:
                    array.append(options['preprocess'].wv[word])
                except KeyError:
                    array.append(np.zeros((options['length'],)))

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
    def preprocess_scaler(array: np.ndarray, **options) -> np.ndarray:

        if array.shape != ():
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        else:
            array = options['preprocess'].transform(array.reshape(-1, 1))[0]

        return array

    @staticmethod
    def preprocess_classification(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def preprocess_regression(array: np.ndarray, **options) -> np.ndarray:

        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            array = options['preprocess'].transform(array.reshape(-1, 1))[0]

        return array

    @staticmethod
    def preprocess_segmentation(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def preprocess_text_segmentation(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def preprocess_timeseries(array: np.ndarray, **options) -> np.ndarray:

        if options['scaler'] not in [LayerScalerImageChoice.no_scaler, None]:
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        return array

    @staticmethod
    def preprocess_object_detection(array: list, **options):

        return array

    @staticmethod
    def preprocess_raw(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def get_y_true(options, output_id):
        method_name = 'get_y_true'
        try:
            if not options.data.use_generator:
                y_true = options.Y.get('val').get(f"{output_id}")
            else:
                y_true = []
                for _, y_val in options.dataset['val'].batch(1):
                    y_true.extend(y_val.get(f'{output_id}').numpy())
                y_true = np.array(y_true)
            return y_true
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def get_yolo_y_true(options, dataset_path):
        method_name = 'get_yolo_y_true'
        try:
            y_true = {}
            bb = []
            model_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]
            for index in range(len(options.dataframe['val'])):
                image_path = os.path.join(
                    dataset_path, options.dataframe['val'][options.dataframe['val'].columns[0]][index])
                img = Image.open(image_path)
                real_size = img.size
                scale_w = int(model_size[0] / real_size[0])
                scale_h = int(model_size[1] / real_size[1])
                coord = options.dataframe.get('val').iloc[index, 1].split(' ')
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in coord])
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                classes_gt = to_categorical(
                    classes_gt, num_classes=len(options.data.outputs.get(2).classes_names)
                )
                bboxes_gt = np.concatenate(
                    [bboxes_gt[:, 1:2] * scale_h, bboxes_gt[:, 0:1] * scale_w,
                     bboxes_gt[:, 3:4] * scale_h, bboxes_gt[:, 2:3] * scale_w], axis=-1)
                conf_gt = np.expand_dims(np.ones(len(bboxes_gt)), axis=-1)
                _bb = np.concatenate([bboxes_gt, conf_gt, classes_gt], axis=-1)
                bb.append(_bb)
            for channel in range(len(options.data.outputs.keys())):
                y_true[channel] = bb
            return y_true
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def get_yolo_y_pred(array, options, sensitivity: float = 0.15, threashold: float = 0.1):
        method_name = 'get_yolo_y_pred'
        try:
            y_pred = {}
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            for i, box_array in enumerate(array):
                channel_boxes = []
                for ex in box_array:
                    boxes = CreateArray().get_predict_boxes(
                        array=np.expand_dims(ex, axis=0),
                        name_classes=name_classes,
                        bb_size=i,
                        sensitivity=sensitivity,
                        threashold=threashold
                    )
                    channel_boxes.append(boxes)
                y_pred[i] = channel_boxes
            return y_pred
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            x_val = None
            inverse_x_val = None
            if options.data.architecture in [ArchitectureChoice.Basic, ArchitectureChoice.ImageClassification,
                                             ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                             ArchitectureChoice.TextClassification,
                                             ArchitectureChoice.AudioClassification,
                                             ArchitectureChoice.VideoClassification,
                                             ArchitectureChoice.DataframeClassification,
                                             ArchitectureChoice.DataframeRegression, ArchitectureChoice.Timeseries,
                                             ArchitectureChoice.TimeseriesTrend]:
                if options.data.group == DatasetGroupChoice.keras:
                    x_val = options.X.get("val")
                dataframe = False
                for inp in options.data.inputs.keys():
                    if options.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                        dataframe = True
                        break
                ts = False
                for out in options.data.outputs.keys():
                    if options.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries or \
                            options.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
                        ts = True
                        break
                if dataframe and not options.data.use_generator:
                    x_val = options.X.get("val")

                elif dataframe and options.data.use_generator:
                    x_val = {}
                    for inp in options.dataset['val'].keys():
                        x_val[inp] = []
                        for x_val_, _ in options.dataset['val'].batch(1):
                            x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                        x_val[inp] = np.array(x_val[inp])
                else:
                    pass

                if ts:
                    inverse_x_val = {}
                    for input in x_val.keys():
                        preprocess_dict = options.preprocessing.preprocessing.get(int(input))
                        inverse_x = np.zeros_like(x_val.get(input)[:, :, 0:1])
                        for i, column in enumerate(preprocess_dict.keys()):
                            if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                _options = {
                                    int(input): {
                                        column: x_val.get(input)[:, :, i]
                                    }
                                }
                                inverse_col = np.expand_dims(
                                    options.preprocessing.inverse_data(_options).get(int(input)).get(column), axis=-1)
                            else:
                                inverse_col = x_val.get(input)[:, :, i:i + 1]
                            inverse_x = np.concatenate([inverse_x, inverse_col], axis=-1)
                        inverse_x_val[input] = inverse_x[:, :, 1:]
            return x_val, inverse_x_val
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_results(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                            threashold=0.1) -> dict:
        method_name = 'postprocess_results'
        try:
            # print('\npostprocess_results', options.data)
            x_array, inverse_x_array = CreateArray().get_x_array(options)
            return_data = {}

            if options.data.architecture in [ArchitectureChoice.Basic, ArchitectureChoice.ImageClassification,
                                             ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                             ArchitectureChoice.TextClassification,
                                             ArchitectureChoice.AudioClassification,
                                             ArchitectureChoice.VideoClassification,
                                             ArchitectureChoice.DataframeClassification,
                                             ArchitectureChoice.DataframeRegression, ArchitectureChoice.Timeseries,
                                             ArchitectureChoice.TimeseriesTrend]:
                for i, output_id in enumerate(options.data.outputs.keys()):
                    true_array = CreateArray().get_y_true(options, output_id)
                    if len(options.data.outputs.keys()) > 1:
                        postprocess_array = array[i]
                    else:
                        postprocess_array = array
                    example_idx = CreateArray().prepare_example_idx_to_show(
                        array=postprocess_array,
                        true_array=true_array,
                        options=options,
                        output=output_id,
                        count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
                    )
                    if options.data.outputs[output_id].task == LayerOutputTypeChoice.Classification:
                        return_data[output_id] = []
                        _id = 1
                        for idx in example_idx:
                            input_id = list(options.data.inputs.keys())[0]
                            source = CreateArray().postprocess_initial_source(
                                options=options,
                                input_id=input_id,
                                save_id=_id,
                                example_id=idx,
                                dataset_path=dataset_path,
                                preset_path=save_path,
                                x_array=None if not x_array else x_array.get(f"{input_id}"),
                                inverse_x_array=None if not inverse_x_array else inverse_x_array.get(f"{input_id}"),
                                return_mode='deploy'
                            )
                            actual_value, predict_values = CreateArray().postprocess_classification(
                                predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                                true_array=true_array[idx],
                                options=options.data.outputs[output_id],
                                return_mode='deploy'
                            )

                            return_data[output_id].append(
                                {
                                    "source": source,
                                    "actual": actual_value,
                                    "data": predict_values[0]
                                }
                            )
                            _id += 1

                    elif options.data.outputs[output_id].task == LayerOutputTypeChoice.TimeseriesTrend:
                        return_data[output_id] = {}
                        # TODO: считаетм что инпут один
                        input_id = list(options.data.inputs.keys())[0]
                        inp_col_id = []
                        for j, out_col in enumerate(options.data.columns.get(output_id).keys()):
                            for k, inp_col in enumerate(options.data.columns.get(input_id).keys()):
                                if out_col.split('_', 1)[-1] == inp_col.split('_', 1)[-1]:
                                    inp_col_id.append((k, inp_col, j, out_col))
                                    break
                        preprocess = options.preprocessing.preprocessing.get(output_id)
                        for channel in inp_col_id:
                            return_data[output_id][channel[3]] = []
                            for idx in example_idx:
                                if type(preprocess.get(channel[3])).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                    inp_options = {int(output_id): {
                                        channel[3]: options.X.get('val').get(f"{input_id}")[idx,
                                                    channel[0]:channel[0] + 1]}
                                    }
                                    inverse_true = options.preprocessing.inverse_data(inp_options).get(output_id).get(
                                        channel[3])
                                    inverse_true = inverse_true.squeeze().astype('float').tolist()
                                else:
                                    inverse_true = options.X.get('val').get(f"{input_id}")[
                                                   idx, channel[0]:channel[0] + 1].squeeze().astype('float').tolist()
                                actual_value, predict_values = CreateArray().postprocess_classification(
                                    predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                                    true_array=true_array[idx],
                                    options=options.data.outputs[output_id],
                                    return_mode='deploy'
                                )
                                button_save_path = os.path.join(
                                    save_path, f"ts_trend_button_channel_{channel[2]}_image_{idx}.jpg")
                                # plt.plot(inverse_true)
                                # plt.savefig(button_save_path)
                                # plt.close()
                                return_data[output_id][channel[3]].append(
                                    {
                                        "button_link": button_save_path,
                                        "data": [inverse_true, predict_values]
                                    }
                                )

                    elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Segmentation:
                        return_data[output_id] = []
                        data = []
                        for j, cls in enumerate(options.data.outputs.get(output_id).classes_names):
                            data.append((cls, options.data.outputs.get(output_id).classes_colors[j].as_rgb_tuple()))
                        for idx in example_idx:
                            input_id = list(options.data.inputs.keys())[0]
                            colors = [color.as_rgb_tuple() for color in
                                      options.data.outputs.get(output_id).classes_colors]
                            return_data[output_id].append(
                                {
                                    "source": CreateArray().postprocess_initial_source(
                                        options=options,
                                        input_id=input_id,
                                        save_id=idx,
                                        example_id=idx,
                                        dataset_path=dataset_path,
                                        preset_path=save_path,
                                        x_array=None if not x_array else x_array.get(f"{input_id}"),
                                        inverse_x_array=None if not inverse_x_array else inverse_x_array.get(
                                            f"{input_id}"),
                                        return_mode='deploy'
                                    ),
                                    "segment": CreateArray().postprocess_segmentation(
                                        predict_array=array[idx],
                                        true_array=None,
                                        options=options.data.outputs.get(output_id),
                                        output_id=output_id,
                                        image_id=idx,
                                        save_path=save_path,
                                        return_mode='deploy'
                                    ),
                                    "data": data
                                }
                            )

                    elif options.data.outputs[output_id].task == LayerOutputTypeChoice.TextSegmentation:
                        return_data[output_id] = {
                            "color_map": None,
                            "data": []
                        }
                        output_column = list(options.instructions.get(output_id).keys())[0]
                        for idx in example_idx:
                            source, segment, colors = CreateArray().postprocess_text_segmentation(
                                pred_array=postprocess_array[idx],
                                options=options.data.outputs[output_id],
                                dataframe=options.dataframe.get("val"),
                                example_id=idx,
                                dataset_params=options.instructions.get(output_id).get(output_column),
                                return_mode='deploy'
                            )
                            return_data[output_id]["data"].append(
                                {
                                    "source": source,
                                    "format": segment,
                                    # "data": colors
                                }
                            )
                        return_data[output_id]["color_map"] = colors

                    elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Timeseries:
                        return_data[output_id] = []
                        preprocess = options.preprocessing.preprocessing.get(output_id)
                        for idx in example_idx:
                            data = {
                                'source': {},
                                'predict': {}
                            }
                            for inp in options.data.inputs.keys():
                                for k, inp_col in enumerate(options.data.columns.get(inp).keys()):
                                    data['source'][inp_col.split('_', 1)[-1]] = \
                                        CreateArray._round_list(list(inverse_x_array[f"{inp}"][idx][:, k]))

                            for ch, channel in enumerate(options.data.columns.get(output_id).keys()):
                                if type(preprocess.get(channel)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                    inp_options = {output_id: {
                                        channel: options.Y.get('val').get(f"{output_id}")[idx, :, ch:ch + 1]}
                                    }
                                    inverse_true = options.preprocessing.inverse_data(inp_options).get(output_id).get(
                                        channel)
                                    inverse_true = CreateArray()._round_list(
                                        inverse_true.squeeze().astype('float').tolist())
                                    out_options = {int(output_id): {
                                        channel: array[idx, :, ch:ch + 1].reshape(-1, 1)}
                                    }
                                    inverse_pred = options.preprocessing.inverse_data(out_options).get(output_id).get(
                                        channel)
                                    inverse_pred = CreateArray()._round_list(
                                        inverse_pred.squeeze().astype('float').tolist())
                                else:
                                    inverse_true = options.Y.get('val').get(f"{output_id}")[
                                                   idx, :, ch:ch + 1].squeeze().astype('float').tolist()
                                    inverse_pred = array[idx, :, ch:ch + 1].squeeze().astype('float').tolist()
                                data['predict'][channel.split('_', 1)[-1]] = [inverse_true, inverse_pred]
                            return_data[output_id].append(data)

                    elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Regression:
                        return_data[output_id] = {
                            'preset': [],
                            'label': []
                        }
                        source_col = []
                        for inp in options.data.inputs.keys():
                            source_col.extend(list(options.data.columns.get(inp).keys()))
                        preprocess = options.preprocessing.preprocessing.get(output_id)
                        for idx in example_idx:
                            row_list = []
                            for inp_col in source_col:
                                row_list.append(f"{options.dataframe.get('val')[inp_col][idx]}")
                            return_data[output_id]['preset'].append(row_list)
                            for i, col in enumerate(list(options.data.columns.get(output_id).keys())):
                                if type(preprocess.get(col)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                    _options = {int(output_id): {col: array[idx, i:i + 1].reshape(-1, 1)}}
                                    inverse_col = options.preprocessing.inverse_data(_options).get(output_id).get(col)
                                    inverse_col = inverse_col.squeeze().astype('float').tolist()
                                else:
                                    inverse_col = array[idx, i:i + 1].astype('float').tolist()
                            return_data[output_id]['label'].append([str(inverse_col)])

                    else:
                        return_data[output_id] = []

            elif options.data.architecture in [ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4]:
                y_true = CreateArray().get_yolo_y_true(options, dataset_path)
                y_pred = CreateArray().get_yolo_y_pred(array, options, sensitivity=sensitivity, threashold=threashold)
                name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
                hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
                colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
                image_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]

                example_idx, bb = CreateArray().prepare_yolo_example_idx_to_show(
                    array=y_pred,
                    true_array=y_true,
                    name_classes=options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names,
                    box_channel=None,
                    count=int(len(y_pred[0]) * DEPLOY_PRESET_PERCENT / 100),
                    choice_type='best',
                    sensitivity=sensitivity,
                    get_optimal_channel=True
                )
                return_data[bb] = []
                for ex in example_idx:
                    img_path = os.path.join(dataset_path, options.dataframe['val'].iat[ex, 0])
                    img = Image.open(img_path)
                    img = img.resize(image_size, Image.BICUBIC)
                    img = img.convert('RGB')
                    source = os.path.join(save_path, f"deploy_od_initial_data_{ex}_box_{bb}.webp")
                    img.save(source, 'webp')
                    save_predict_path, _ = CreateArray().plot_boxes(
                        true_bb=y_true[bb][ex],
                        pred_bb=y_pred[bb][ex],
                        img_path=img_path,
                        name_classes=name_classes,
                        colors=colors,
                        image_id=ex,
                        add_only_true=False,
                        plot_true=False,
                        image_size=image_size,
                        save_path=save_path,
                        return_mode='deploy'
                    )
                    return_data[bb].append(
                        {
                            "source": source,
                            "predict": save_predict_path
                        }
                    )
            else:
                return_data = {}
            return return_data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   save_id: int = None, x_array=None, inverse_x_array=None, return_mode='deploy',
                                   max_lenth: int = 50, templates: list = None):
        method_name = 'postprocess_initial_source'
        try:
            column_idx = []
            input_task = options.data.inputs.get(input_id).task
            if options.data.group != DatasetGroupChoice.keras:
                for inp in options.data.inputs.keys():
                    if options.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                        input_task = LayerInputTypeChoice.Dataframe
                    for column_name in options.dataframe.get('val').columns:
                        if column_name.split('_')[0] == f"{inp}":
                            column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
                if input_task == LayerInputTypeChoice.Text or input_task == LayerInputTypeChoice.Dataframe:
                    initial_file_path = ""
                else:
                    initial_file_path = os.path.join(dataset_path,
                                                     options.dataframe.get('val').iat[example_id, column_idx[0]])
                if not save_id:
                    return str(os.path.abspath(initial_file_path))
            else:
                initial_file_path = ""

            data = []
            data_type = ""
            source = ""

            if input_task == LayerInputTypeChoice.Image:
                if options.data.group != DatasetGroupChoice.keras:
                    img = Image.open(initial_file_path)
                    img = img.resize(
                        options.data.inputs.get(input_id).shape[0:2][::-1],
                        Image.ANTIALIAS
                    )
                else:
                    img = image.array_to_img(x_array[example_id])
                img = img.convert('RGB')
                source = os.path.join(preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp")
                img.save(source, 'webp')
                if return_mode == 'callback':
                    data_type = LayerInputTypeChoice.Image.name
                    data = [
                        {
                            "title": "Изображение",
                            "value": source,
                            "color_mark": None
                        }
                    ]

            elif input_task == LayerInputTypeChoice.Text:
                regression_task = False
                for out in options.data.outputs.keys():
                    if options.data.outputs.get(out).task == LayerOutputTypeChoice.Regression:
                        regression_task = True
                for column in column_idx:
                    source = options.dataframe.get('val').iat[example_id, column]
                    if return_mode == 'deploy':
                        break
                    if return_mode == 'callback':
                        data_type = LayerInputTypeChoice.Text.name
                        title = "Текст"
                        if regression_task:
                            title = list(options.dataframe.get('val').columns)[column].split("_", 1)[-1]
                        data = [
                            {
                                "title": title,
                                "value": source,
                                "color_mark": None
                            }
                        ]

            elif input_task == LayerInputTypeChoice.Video:
                clip = moviepy_editor.VideoFileClip(initial_file_path)
                source = os.path.join(preset_path, f"initial_data_video_{save_id}_input_{input_id}.webm")
                clip.write_videofile(source)
                if return_mode == 'callback':
                    data_type = LayerInputTypeChoice.Video.name
                    data = [
                        {
                            "title": "Видео",
                            "value": source,
                            "color_mark": None
                        }
                    ]

            elif input_task == LayerInputTypeChoice.Audio:
                source = os.path.join(preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webm")
                AudioSegment.from_file(initial_file_path).export(source, format="webm")
                if return_mode == 'callback':
                    data_type = LayerInputTypeChoice.Audio.name
                    data = [
                        {
                            "title": "Аудио",
                            "value": source,
                            "color_mark": None
                        }
                    ]

            elif input_task == LayerInputTypeChoice.Dataframe:
                time_series_choice = False
                for out in options.data.outputs.keys():
                    if options.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries or \
                            options.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
                        time_series_choice = True
                        break
                if time_series_choice:
                    graphics_data = []
                    names = ""
                    multi = False
                    if return_mode == 'callback':
                        for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                            multi = True if i > 0 else False
                            names += f"«{channel.split('_', 1)[-1]}», "
                            lenth = len(inverse_x_array) if len(inverse_x_array) < max_lenth else max_lenth
                            graphics_data.append(
                                templates[1](
                                    _id=i + 1,
                                    _type='graphic',
                                    graph_name=f"График канала «{channel.split('_', 1)[-1]}»",
                                    short_name=f"«{channel.split('_', 1)[-1]}»",
                                    x_label="Время",
                                    y_label="Значение",
                                    plot_data=[
                                        templates[0](
                                            label="Исходное значение",
                                            x=np.arange(inverse_x_array[example_id].shape[-2]).astype('int').tolist()[
                                              -lenth:],
                                            y=inverse_x_array[example_id][:, i].astype('float').tolist()[-lenth:]
                                        )
                                    ],
                                )
                            )
                        data_type = "graphic"
                        data = [
                            {
                                "title": f"График{'и' if multi else ''} по канал{'ам' if multi else 'у'} {names[:-2]}",
                                "value": graphics_data,
                                "color_mark": None
                            }
                        ]
                else:
                    data_type = "str"
                    source = []
                    # for inp in options.data.inputs.keys():
                    for col_name in options.data.columns.get(input_id).keys():
                        value = options.dataframe.get('val')[col_name].to_list()[example_id]
                        # source.append((col_name, value))
                        if return_mode == 'deploy':
                            source.append(value)
                        if return_mode == 'callback':
                            data.append(
                                {
                                    "title": col_name.split("_", 1)[-1],
                                    "value": value,
                                    "color_mark": None
                                }
                            )

            else:
                pass

            if return_mode == 'deploy':
                return source
            if return_mode == 'callback':
                return data, data_type.lower()
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
        method_name = 'dice_coef'
        try:
            axis = tuple(np.arange(1, len(y_true.shape))) if batch_mode else None
            intersection = np.sum(y_true * y_pred, axis=axis)
            union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
            return (2.0 * intersection + smooth) / (union + smooth)
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def sort_dict(dict_to_sort: dict, mode: BalanceSortedChoice = BalanceSortedChoice.alphabetic):
        method_name = 'sort_dict'
        try:
            if mode == BalanceSortedChoice.alphabetic:
                sorted_keys = sorted(dict_to_sort)
                sorted_values = []
                for w in sorted_keys:
                    sorted_values.append(dict_to_sort[w])
                return tuple(sorted_keys), tuple(sorted_values)
            elif mode == BalanceSortedChoice.ascending:
                sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get)
                sorted_values = []
                for w in sorted_keys:
                    sorted_values.append(dict_to_sort[w])
                return tuple(sorted_keys), tuple(sorted_values)
            elif mode == BalanceSortedChoice.descending:
                sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)
                sorted_values = []
                for w in sorted_keys:
                    sorted_values.append(dict_to_sort[w])
                return tuple(sorted_keys), tuple(sorted_values)
            else:
                return tuple(dict_to_sort.keys()), tuple(dict_to_sort.values())
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: str = "best", seed_idx: list = None) -> dict:
        method_name = 'prepare_example_idx_to_show'
        try:
            example_idx = []
            encoding = options.data.outputs.get(output).encoding
            task = options.data.outputs.get(output).task
            if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                    if array.shape[-1] == true_array.shape[-1] and encoding == \
                            LayerEncodingChoice.ohe and true_array.shape[-1] > 1:
                        classes = np.argmax(true_array, axis=-1)
                    elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] > 1:
                        classes = copy.deepcopy(true_array)
                    elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[
                        -1] == 1:
                        classes = copy.deepcopy(true_array)
                    else:
                        classes = copy.deepcopy(true_array)
                    class_idx = {}
                    for _id in range(options.data.outputs.get(output).num_classes):
                        class_idx[_id] = {}
                    for i, pred in enumerate(array):
                        class_idx[classes[i]][i] = pred[classes[i]]
                    for _id in range(options.data.outputs.get(output).num_classes):
                        class_idx[_id] = list(CreateArray().sort_dict(
                            class_idx[_id], mode=BalanceSortedChoice.ascending)[0])

                    num_ex = copy.deepcopy(count)
                    while num_ex:
                        key = np.random.choice(list(class_idx.keys()))
                        if choice_type == ExampleChoiceTypeChoice.best:
                            example_idx.append(class_idx[key][-1])
                            class_idx[key].pop(-1)
                        if choice_type == ExampleChoiceTypeChoice.worst:
                            example_idx.append(class_idx[key][0])
                            class_idx[key].pop(0)
                        num_ex -= 1

                elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
                    if encoding == LayerEncodingChoice.ohe:
                        array = to_categorical(
                            np.argmax(array, axis=-1),
                            num_classes=options.data.outputs.get(output).num_classes
                        )
                    if encoding == LayerEncodingChoice.multi:
                        array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1, 0)
                    dice_val = CreateArray().dice_coef(true_array, array, batch_mode=True)
                    dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
                    if choice_type == ExampleChoiceTypeChoice.best:
                        example_idx, _ = CreateArray().sort_dict(dice_dict, mode=BalanceSortedChoice.descending)
                        example_idx = example_idx[:count]
                    if choice_type == ExampleChoiceTypeChoice.worst:
                        example_idx, _ = CreateArray().sort_dict(dice_dict, mode=BalanceSortedChoice.ascending)
                        example_idx = example_idx[:count]

                elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                    delta = np.abs(true_array - array) * 100 / true_array
                    while len(delta.shape) != 1:
                        delta = np.mean(delta, axis=-1)
                    delta_dict = dict(zip(np.arange(0, len(delta)), delta))
                    if choice_type == ExampleChoiceTypeChoice.best:
                        example_idx, _ = CreateArray().sort_dict(delta_dict, mode=BalanceSortedChoice.ascending)
                        example_idx = example_idx[:count]
                    if choice_type == ExampleChoiceTypeChoice.worst:
                        example_idx, _ = CreateArray().sort_dict(delta_dict, mode=BalanceSortedChoice.descending)
                        example_idx = example_idx[:count]

                else:
                    pass

            elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
                example_idx = seed_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.random:
                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                    true_id = []
                    false_id = []
                    for i, ex in enumerate(true_array):
                        if np.argmax(ex, axis=-1) == np.argmax(array[i], axis=-1):
                            true_id.append(i)
                        else:
                            false_id.append(i)
                    np.random.shuffle(true_id)
                    np.random.shuffle(false_id)
                    true_false_dict = {'true': true_id, 'false': false_id}

                    for _ in range(count):
                        if true_false_dict.get('true') and true_false_dict.get('false'):
                            key = np.random.choice(list(true_false_dict.keys()))
                        elif true_false_dict.get('true') and not true_false_dict.get('false'):
                            key = 'true'
                        else:
                            key = 'false'
                        example_idx.append(true_false_dict.get(key)[0])
                        true_false_dict.get(key).pop(0)
                    np.random.shuffle(example_idx)

                elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
                    if encoding == LayerEncodingChoice.ohe:
                        array = to_categorical(
                            np.argmax(array, axis=-1),
                            num_classes=options.data.outputs.get(output).num_classes
                        )
                    if encoding == LayerEncodingChoice.multi:
                        array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1, 0)
                    dice_val = CreateArray().dice_coef(true_array, array, batch_mode=True)

                    true_id = []
                    false_id = []
                    for i, ex in enumerate(dice_val):
                        if ex >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100:
                            true_id.append(i)
                        else:
                            false_id.append(i)
                    np.random.shuffle(true_id)
                    np.random.shuffle(false_id)
                    true_false_dict = {'true': true_id, 'false': false_id}

                    for _ in range(count):
                        if true_false_dict.get('true') and true_false_dict.get('false'):
                            key = np.random.choice(list(true_false_dict.keys()))
                        elif true_false_dict.get('true') and not true_false_dict.get('false'):
                            key = 'true'
                        else:
                            key = 'false'
                        example_idx.append(true_false_dict.get(key)[0])
                        true_false_dict.get(key).pop(0)
                    np.random.shuffle(example_idx)

                elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                    delta = np.abs(true_array - array) * 100 / true_array
                    while len(delta.shape) != 1:
                        delta = np.mean(delta, axis=-1)
                    true_id = []
                    false_id = []
                    for i, ex in enumerate(delta):
                        if ex >= CALLBACK_REGRESSION_TREASHOLD_VALUE:
                            true_id.append(i)
                        else:
                            false_id.append(i)
                    np.random.shuffle(true_id)
                    np.random.shuffle(false_id)
                    true_false_dict = {'true': true_id, 'false': false_id}

                    for _ in range(count):
                        if true_false_dict.get('true') and true_false_dict.get('false'):
                            key = np.random.choice(list(true_false_dict.keys()))
                        elif true_false_dict.get('true') and not true_false_dict.get('false'):
                            key = 'true'
                        else:
                            key = 'false'
                        example_idx.append(true_false_dict.get(key)[0])
                        true_false_dict.get(key).pop(0)
                    np.random.shuffle(example_idx)

                else:
                    example_idx = np.random.randint(0, len(true_array), count)

            else:
                pass
            return example_idx
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def prepare_yolo_example_idx_to_show(array: dict, true_array: dict, name_classes: list, box_channel: int,
                                         count: int, choice_type: str = "best", seed_idx: list = None,
                                         sensitivity: float = 0.25, get_optimal_channel=False):
        method_name = 'prepare_yolo_example_idx_to_show'
        try:
            # print('\nprepare_yolo_example_idx_to_show')
            if get_optimal_channel:
                channel_stat = []
                for channel in range(3):
                    total_metric = 0
                    for example in range(len(array.get(channel))):
                        total_metric += CreateArray().get_yolo_example_statistic(
                            true_bb=true_array.get(channel)[example],
                            pred_bb=array.get(channel)[example],
                            name_classes=name_classes,
                            sensitivity=sensitivity
                        )['total_stat']['total_metric']
                    channel_stat.append(total_metric / len(array.get(channel)))
                box_channel = int(np.argmax(channel_stat, axis=-1))

            if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
                stat = []
                for example in range(len(array.get(box_channel))):
                    stat.append(
                        CreateArray().get_yolo_example_statistic(
                            true_bb=true_array.get(box_channel)[example],
                            pred_bb=array.get(box_channel)[example],
                            name_classes=name_classes,
                            sensitivity=sensitivity
                        )['total_stat']['total_metric']
                    )
                stat_dict = dict(zip(np.arange(0, len(stat)), stat))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = CreateArray().sort_dict(stat_dict, mode=BalanceSortedChoice.descending)
                    example_idx = example_idx[:count]
                else:
                    example_idx, _ = CreateArray().sort_dict(stat_dict, mode=BalanceSortedChoice.ascending)
                    example_idx = example_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.seed:
                example_idx = seed_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.random:
                true_false_dict = {'true': [], 'false': []}
                for i, example in enumerate(array.get(box_channel)):
                    ex_stat = CreateArray().get_yolo_example_statistic(
                        true_bb=true_array.get(box_channel)[example],
                        pred_bb=array.get(box_channel)[example],
                        name_classes=name_classes,
                        sensitivity=sensitivity
                    )['total_stat']['total_metric']
                    if ex_stat > 0.7:
                        true_false_dict['true'].append(i)
                    else:
                        true_false_dict['false'].append(i)
                example_idx = []
                for _ in range(count):
                    if true_false_dict.get('true') and true_false_dict.get('false'):
                        key = np.random.choice(list(true_false_dict.keys()))
                    elif true_false_dict.get('true') and not true_false_dict.get('false'):
                        key = 'true'
                    else:
                        key = 'false'
                    example_idx.append(true_false_dict.get(key)[0])
                    true_false_dict.get(key).pop(0)
                np.random.shuffle(example_idx)
                # example_idx = np.random.randint(0, len(true_array.get(box_channel)), count)
            else:
                example_idx = np.random.randint(0, len(true_array.get(box_channel)), count)
            return example_idx, box_channel
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_classification(predict_array: np.ndarray, true_array: np.ndarray, options: DatasetOutputsData,
                                   show_stat: bool = False, return_mode='deploy'):
        method_name = 'postprocess_classification'
        try:
            labels = options.classes_names
            ohe = True if options.encoding == LayerEncodingChoice.ohe else False
            actual_value = np.argmax(true_array, axis=-1) if ohe else true_array
            data = {
                "y_true": {},
                "y_pred": {},
                "stat": {}
            }
            if return_mode == 'deploy':
                labels_from_array = []
                for class_idx in predict_array:
                    class_dist = sorted(class_idx, reverse=True)
                    labels_dist = []
                    for j in class_dist:
                        labels_dist.append((labels[list(class_idx).index(j)], round(float(j) * 100, 1)))
                    labels_from_array.append(labels_dist)
                return labels[actual_value], labels_from_array

            elif return_mode == 'callback':
                data["y_true"] = {
                    "type": "str",
                    "data": [
                        {
                            "title": "Класс",
                            "value": labels[actual_value],
                            "color_mark": None
                        }
                    ]
                }
                if labels[actual_value] == labels[np.argmax(predict_array)]:
                    color_mark = 'success'
                else:
                    color_mark = 'wrong'
                data["y_pred"] = {
                    "type": "str",
                    "data": [
                        {
                            "title": "Класс",
                            "value": labels[np.argmax(predict_array)],
                            "color_mark": color_mark
                        }
                    ]
                }
                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    for i, val in enumerate(predict_array):
                        if val == max(predict_array) and labels[i] == labels[actual_value]:
                            class_color_mark = "success"
                        elif val == max(predict_array) and labels[i] != labels[actual_value]:
                            class_color_mark = "wrong"
                        else:
                            class_color_mark = None
                        data["stat"]["data"].append(
                            {
                                'title': labels[i],
                                'value': f"{round(val * 100, 1)}%",
                                'color_mark': class_color_mark
                            }
                        )
                return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_segmentation(predict_array: np.ndarray, true_array: np.ndarray, options: DatasetOutputsData,
                                 output_id: int, image_id: int, save_path: str, colors: list = None,
                                 return_mode='deploy', show_stat: bool = False):
        method_name = 'postprocess_segmentation'
        try:
            data = {
                "y_true": {},
                "y_pred": {},
                "stat": {}
            }
            if return_mode == 'deploy':
                array = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(options.classes_colors):
                    array = np.where(
                        array == i * 512,
                        np.array(color.as_rgb_tuple()),
                        array
                    )
                array = array.astype("uint8")
                img_save_path = os.path.join(
                    save_path,
                    f"image_segmentation_postprocessing_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(img_save_path, array)
                return img_save_path

            if return_mode == 'callback':
                y_true = np.expand_dims(np.argmax(true_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(colors):
                    y_true = np.where(y_true == i * 512, np.array(color), y_true)
                y_true = y_true.astype("uint8")
                y_true_save_path = os.path.join(
                    save_path,
                    f"true_segmentation_data_image_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(y_true_save_path, y_true)

                data["y_true"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": y_true_save_path,
                            "color_mark": None
                        }
                    ]
                }

                y_pred = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
                for i, color in enumerate(colors):
                    y_pred = np.where(y_pred == i * 512, np.array(color), y_pred)
                y_pred = y_pred.astype("uint8")
                y_pred_save_path = os.path.join(
                    save_path,
                    f"predict_segmentation_data_image_{image_id}_output_{output_id}.webp"
                )
                matplotlib.image.imsave(y_pred_save_path, y_pred)
                data["y_pred"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": y_pred_save_path,
                            "color_mark": None
                        }
                    ]
                }
                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    y_true = np.array(true_array).astype('int')
                    y_pred = to_categorical(np.argmax(predict_array, axis=-1), options.num_classes).astype('int')
                    count = 0
                    mean_val = 0
                    for idx, cls in enumerate(options.classes_names):
                        dice_val = np.round(
                            CreateArray().dice_coef(y_true[:, :, idx], y_pred[:, :, idx], batch_mode=False) * 100, 1)
                        count += 1
                        mean_val += dice_val
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': f"{dice_val}%",
                                'color_mark': 'success' if dice_val >= 90 else 'wrong'
                            }
                        )
                    data["stat"]["data"].insert(
                        0,
                        {
                            'title': "Средняя точность",
                            'value': f"{round(mean_val / count, 2)}%",
                            'color_mark': 'success' if mean_val / count >= 90 else 'wrong'
                        }
                    )
                return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_text_segmentation(pred_array: np.ndarray, options: DatasetOutputsData, dataframe: DataFrame,
                                      dataset_params: dict, example_id: int, return_mode='deploy',
                                      class_colors: list = None,
                                      show_stat: bool = False, true_array: np.ndarray = None):
        method_name = 'postprocess_text_segmentation'
        try:

            def add_tags_to_word(word: str, tag: str):
                if tag:
                    for t in tag:
                        word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
                    return word
                else:
                    return f"<p1>{word}</p1>"

            def reformat_tags(y_array: np.ndarray, tag_list: list,  # classes_names: dict, colors: dict,
                              sensitivity: float = 0.9):
                norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
                reformat_tags = []
                for word_tag in norm_array:
                    if np.sum(word_tag) == 0:
                        reformat_tags.append(None)
                    else:
                        mix_tag = []
                        for i, tag in enumerate(word_tag):
                            if tag == 1:
                                mix_tag.append(tag_list[i])
                        reformat_tags.append(mix_tag)
                return reformat_tags

            def text_colorization(text: str, label_array: np.ndarray, tag_list: list, class_names: dict, colors: dict):
                text = text.split(" ")
                labels = reformat_tags(label_array, tag_list)
                colored_text = []
                for i, word in enumerate(text):
                    colored_text.append(add_tags_to_word(word, labels[i]))
                return ' '.join(colored_text)

            # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
            #  на сегментацию текста на несколько входов то придется искать решения

            classes_names = {}
            dataset_tags = dataset_params.get("open_tags").split()
            colors = {}
            if options.classes_colors and return_mode == 'deploy':
                for i, name in enumerate(dataset_tags):
                    colors[name] = options.classes_colors[i].as_rgb_tuple()
                    classes_names[name] = options.classes_names[i]
            elif class_colors and return_mode == 'callback':
                for i, name in enumerate(dataset_tags):
                    colors[name] = class_colors[i]
                    classes_names[name] = options.classes_names[i]
            else:
                hsv_tuples = [(x / len(dataset_tags), 1., 1.) for x in range(len(dataset_tags))]
                gen_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                gen_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), gen_colors))
                for i, name in enumerate(dataset_tags):
                    colors[name] = gen_colors[i]
                    classes_names[name] = options.classes_names[i]

            if return_mode == 'deploy':
                initinal_text = dataframe.iat[example_id, 0]
                text_segmentation = text_colorization(
                    text=initinal_text,
                    label_array=pred_array,
                    tag_list=dataset_tags,
                    class_names=classes_names,
                    colors=colors
                )

                data = [('<p1>', '<p1>', (200, 200, 200))]
                for tag in colors.keys():
                    data.append(
                        (tag, classes_names[tag], colors[tag])
                    )
                return initinal_text, text_segmentation, data

            if return_mode == 'callback':
                data = {"y_true": {}, "y_pred": {}, "tags_color": {}, "stat": {}}
                text_for_preparation = dataframe.iat[example_id, 0]
                true_text_segmentation = text_colorization(
                    text=text_for_preparation,
                    label_array=true_array,
                    tag_list=dataset_tags,
                    class_names=classes_names,
                    colors=colors
                )

                data["y_true"] = {
                    "type": "segmented_text",
                    "data": [{"title": "Текст", "value": true_text_segmentation, "color_mark": None}]
                }
                pred_text_segmentation = text_colorization(
                    text=text_for_preparation,
                    label_array=pred_array,
                    tag_list=dataset_tags,
                    class_names=classes_names,
                    colors=colors
                )
                data["y_pred"] = {
                    "type": "segmented_text",
                    "data": [{"title": "Текст", "value": pred_text_segmentation, "color_mark": None}]
                }
                colors_ = {}
                for key, val in colors.items():
                    colors_[key[1:-1]] = val
                data["tags_color"] = colors_

                if show_stat:
                    data["stat"] = {
                        "type": "str",
                        "data": []
                    }
                    y_true = np.array(true_array).astype('int')
                    y_pred = np.where(pred_array >= 0.9, 1., 0.)
                    count = 0
                    mean_val = 0
                    for idx, cls in enumerate(options.classes_names):
                        if np.sum(y_true[:, idx]) == 0 and np.sum(y_pred[:, idx]) == 0:
                            data["stat"]["data"].append({'title': cls, 'value': "-", 'color_mark': None})
                        elif np.sum(y_true[:, idx]) == 0:
                            data["stat"]["data"].append({'title': cls, 'value': "0.0%", 'color_mark': 'wrong'})
                            count += 1
                        else:
                            class_recall = np.sum(y_true[:, idx] * y_pred[:, idx]) * 100 / np.sum(y_true[:, idx])
                            # dice_val = np.round(
                            #     CreateArray().dice_coef(y_true[:, idx], y_pred[:, idx], batch_mode=False) * 100, 1)
                            data["stat"]["data"].append(
                                {
                                    'title': cls,
                                    'value': f"{np.round(class_recall, 1)} %",
                                    'color_mark': 'success' if class_recall >= 90 else 'wrong'
                                }
                            )
                            count += 1
                            mean_val += class_recall
                    if count and mean_val / count >= 90:
                        mean_color_mark = "success"
                        mean_stat = f"{round(mean_val / count, 1)}%"
                    elif count and mean_val / count < 90:
                        mean_color_mark = "wrong"
                        mean_stat = f"{round(mean_val / count, 2)}%"
                    else:
                        mean_color_mark = None
                        mean_stat = '-'
                    data["stat"]["data"].insert(
                        0, {'title': "Средняя точность", 'value': mean_stat, 'color_mark': mean_color_mark}
                    )
                return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_regression(column_names: list, inverse_y_true: np.ndarray, inverse_y_pred: np.ndarray,
                               show_stat: bool = False, return_mode='deploy'):
        method_name = 'postprocess_regression'
        try:
            data = {"y_true": {"type": "str", "data": []}}
            if return_mode == 'deploy':
                source = []
                return source
            else:
                for i, name in enumerate(column_names):
                    data["y_true"]["data"].append(
                        {"title": name.split('_', 1)[-1], "value": f"{inverse_y_true[i]: .2f}", "color_mark": None}
                    )
                deviation = np.abs((inverse_y_pred - inverse_y_true) * 100 / inverse_y_true)
                data["y_pred"] = {
                    "type": "str",
                    "data": []
                }
                for i, name in enumerate(column_names):
                    color_mark = 'success' if deviation[i] < 2 else "wrong"
                    data["y_pred"]["data"].append(
                        {
                            "title": name.split('_', 1)[-1],
                            "value": f"{inverse_y_pred[i]: .2f}",
                            "color_mark": color_mark
                        }
                    )
                if show_stat:
                    data["stat"] = {"type": "str", "data": []}
                    for i, name in enumerate(column_names):
                        color_mark = 'success' if deviation[i] < 2 else "wrong"
                        data["stat"]["data"].append(
                            {
                                'title': f"Отклонение - «{name.split('_', 1)[-1]}»",
                                'value': f"{np.round(deviation[i], 2)} %",
                                'color_mark': color_mark
                            }
                        )
                return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_time_series(options: DatasetData, real_x: np.ndarray, inverse_y_true: np.ndarray,
                                inverse_y_pred: np.ndarray, output_id: int, depth: int, show_stat: bool = False,
                                templates: list = None, max_lenth: int = 50):

        """
        real_x = self.inverse_x_val.get(f"{input}")[example_idx]
        inverse_y_true = self.inverse_y_true.get("val").get(output_id)[example_idx]
        inverse_y_pred = self.inverse_y_pred.get(output_id)[example_idx]
        depth = self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]
        templates = [self._fill_graph_plot_data, self._fill_graph_front_structure]
        """
        method_name = 'postprocess_time_series'
        try:
            data = {"y_true": {}, "y_pred": {}, "stat": {}}
            graphics = []
            _id = 1
            for i, channel in enumerate(options.columns.get(output_id).keys()):
                for inp in options.inputs.keys():
                    for input_column in options.columns.get(inp).keys():
                        if channel.split("_", 1)[-1] == input_column.split("_", 1)[-1]:
                            init_column = list(options.columns.get(inp).keys()).index(input_column)
                            lenth = len(real_x) if len(real_x) < max_lenth else max_lenth
                            x_tr = CreateArray()._round_list(real_x[:, init_column])  # .astype('float'))
                            y_tr = CreateArray()._round_list(inverse_y_true[:, i])
                            y_tr.insert(0, x_tr[-1])
                            y_pr = CreateArray()._round_list(inverse_y_pred[:, i])
                            y_pr.insert(0, x_tr[-1])
                            graphics.append(
                                templates[1](
                                    _id=_id,
                                    _type='graphic',
                                    graph_name=f'График канала «{channel.split("_", 1)[-1]}»',
                                    short_name=f"«{channel.split('_', 1)[-1]}»",
                                    x_label="Время",
                                    y_label="Значение",
                                    plot_data=[
                                        templates[0](
                                            label="Исходное значение",
                                            x=np.arange(real_x.shape[-2]).astype('int').tolist()[-lenth:],
                                            y=x_tr[-lenth:]
                                        ),
                                        templates[0](
                                            label="Истинное значение",
                                            x=np.arange(real_x.shape[-2] - 1, real_x.shape[-2] + depth).astype(
                                                'int').tolist(),
                                            y=y_tr
                                        ),
                                        templates[0](
                                            label="Предсказанное значение",
                                            x=np.arange(real_x.shape[-2] - 1, real_x.shape[-2] + depth).astype(
                                                'int').tolist(),
                                            y=y_pr
                                        ),
                                    ],
                                )
                            )
                            _id += 1
                            break
            data["y_pred"] = {
                "type": "graphic",
                "data": [
                    {
                        "title": "Графики",
                        "value": graphics,
                        "color_mark": None
                    }
                ]
            }
            if show_stat:
                data["stat"] = {
                    "type": "table",
                    "data": []
                }
                for i, channel in enumerate(options.columns.get(output_id).keys()):
                    data["stat"]["data"].append({'title': channel.split("_", 1)[-1], 'value': [], 'color_mark': None})
                    for step in range(inverse_y_true.shape[-2]):
                        deviation = (inverse_y_pred[step, i] - inverse_y_true[step, i]) * 100 / inverse_y_true[step, i]
                        data["stat"]["data"][-1]["value"].append(
                            {
                                "Шаг": f"{step + 1}",
                                "Истина": f"{inverse_y_true[step, i].astype('float'): .2f}",
                                "Предсказание": f"{inverse_y_pred[step, i].astype('float'): .2f}",
                                "Отклонение": {
                                    "value": f"{deviation: .2f} %",
                                    "color_mark": "success" if abs(deviation) < CALLBACK_REGRESSION_TREASHOLD_VALUE
                                    else "wrong"
                                }
                            }
                        )
            return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def postprocess_object_detection(predict_array, true_array, image_path: str, colors: list,
                                     sensitivity: float, image_id: int, save_path: str, show_stat: bool,
                                     name_classes: list, return_mode='deploy', image_size=(416, 416)):
        method_name = 'postprocess_object_detection'
        try:
            data = {
                "y_true": {},
                "y_pred": {},
                "stat": {}
            }
            if return_mode == 'deploy':
                pass

            if return_mode == 'callback':
                save_true_predict_path, _ = CreateArray().plot_boxes(
                    true_bb=true_array, pred_bb=predict_array, img_path=image_path, name_classes=name_classes,
                    colors=colors, image_id=image_id, add_only_true=False, plot_true=True, image_size=image_size,
                    save_path=save_path, return_mode=return_mode
                )

                data["y_true"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": save_true_predict_path,
                            "color_mark": None,
                            "size": "large"
                        }
                    ]
                }

                save_predict_path, _ = CreateArray().plot_boxes(
                    true_bb=true_array, pred_bb=predict_array, img_path=image_path, name_classes=name_classes,
                    colors=colors, image_id=image_id, add_only_true=False, plot_true=False, image_size=image_size,
                    save_path=save_path, return_mode=return_mode
                )

                data["y_pred"] = {
                    "type": "image",
                    "data": [
                        {
                            "title": "Изображение",
                            "value": save_predict_path,
                            "color_mark": None,
                            "size": "large"
                        }
                    ]
                }
                if show_stat:
                    box_stat = CreateArray().get_yolo_example_statistic(
                        true_bb=true_array,
                        pred_bb=predict_array,
                        name_classes=name_classes,
                        sensitivity=sensitivity
                    )
                    data["stat"]["Общая точность"] = {
                        "type": "str",
                        "data": [
                            {
                                "title": "Среднее",
                                "value": f"{np.round(box_stat['total_stat']['total_metric'] * 100, 2)}%",
                                "color_mark": 'success' if box_stat['total_stat']['total_metric'] >= 0.7 else 'wrong'
                            },
                        ]
                    }
                    data["stat"]['Средняя точность'] = {
                        "type": "str",
                        "data": [
                            {
                                "title": "Перекрытие",
                                "value": f"{np.round(box_stat['total_stat']['total_overlap'] * 100, 2)}%",
                                "color_mark": 'success' if box_stat['total_stat']['total_overlap'] >= 0.7 else 'wrong'
                            },
                            {
                                "title": "Объект",
                                "value": f"{np.round(box_stat['total_stat']['total_conf'] * 100, 2)}%",
                                "color_mark": 'success' if box_stat['total_stat']['total_conf'] >= 0.7 else 'wrong'
                            },
                            {
                                "title": "Класс",
                                "value": f"{np.round(box_stat['total_stat']['total_class'] * 100, 2)}%",
                                "color_mark": 'success' if box_stat['total_stat']['total_class'] >= 0.7 else 'wrong'
                            },
                        ]
                    }
                    for class_name in name_classes:
                        mean_overlap = box_stat['class_stat'][class_name]['mean_overlap']
                        mean_conf = box_stat['class_stat'][class_name]['mean_conf']
                        mean_class = box_stat['class_stat'][class_name]['mean_class']
                        data["stat"][f'{class_name}'] = {
                            "type": "str",
                            "data": [
                                {
                                    "title": "Перекрытие",
                                    "value": "-" if mean_overlap is None else f"{np.round(mean_overlap * 100, 2)}%",
                                    "color_mark": 'success' if mean_overlap and mean_overlap >= 0.7 else 'wrong'
                                },
                                {
                                    "title": "Объект",
                                    "value": "-" if mean_conf is None else f"{np.round(mean_conf * 100, 2)}%",
                                    "color_mark": 'success' if mean_conf and mean_conf >= 0.7 else 'wrong'
                                },
                                {
                                    "title": "Класс",
                                    "value": "-" if mean_class is None else f"{np.round(mean_class * 100, 2)}%",
                                    "color_mark": 'success' if mean_class and mean_class >= 0.7 else 'wrong'
                                },
                            ]
                        }
                return data
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def bboxes_iou(boxes1, boxes2):
        method_name = 'bboxes_iou'
        try:
            boxes1 = np.array(boxes1)
            boxes2 = np.array(boxes2)

            boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = np.maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

            return ious
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, sensitivity: float = 0.15):
        """
        :param boxes: list of unscaled bb coordinates
        :param scores: class probability in ohe
        :param sensitivity: float from 0 to 1
        """
        method_name = 'non_max_suppression_fast'
        try:
            if len(boxes) == 0:
                return [], []

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            classes = np.argmax(scores, axis=-1)
            idxs = np.argsort(classes)[..., ::-1]

            mean_iou = []
            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]
                mean_iou.append(overlap)
                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > sensitivity)[0])))

            return pick, mean_iou
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def get_predict_boxes(array, name_classes: list, bb_size: int = 1, sensitivity: float = 0.15,
                          threashold: float = 0.1):
        method_name = 'get_predict_boxes'
        try:
            """
            Boxes for 1 example
            """
            num_classes = len(name_classes)
            anchors = np.array([[10, 13], [16, 30], [33, 23],
                                [30, 61], [62, 45], [59, 119],
                                [116, 90], [156, 198], [373, 326]])

            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

            level_anchor = bb_size
            num_anchors = len(anchors[anchor_mask[level_anchor]])

            grid_shape = array.shape[1:3]

            feats = np.reshape(array, (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5))

            xy_param = feats[..., :2]
            wh_param = feats[..., 2:4]
            conf_param = feats[..., 4:5]
            class_param = feats[..., 5:]

            box_yx = xy_param[..., ::-1].copy()
            box_hw = wh_param[..., ::-1].copy()

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)
            _boxes = np.concatenate([
                box_mins[..., 0:1],
                box_mins[..., 1:2],
                box_maxes[..., 0:1],
                box_maxes[..., 1:2]
            ], axis=-1)

            _boxes_reshape = np.reshape(_boxes, (-1, 4))
            _box_scores = conf_param * class_param
            _box_scores_reshape = np.reshape(_box_scores, (-1, num_classes))
            _class_param_reshape = np.reshape(class_param, (-1, num_classes))
            mask = _box_scores_reshape >= threashold
            _boxes_out = np.zeros_like(_boxes_reshape[0:1])
            _scores_out = np.zeros_like(_box_scores_reshape[0:1])
            _class_param_out = np.zeros_like(_class_param_reshape[0:1])
            for cl in range(num_classes):
                if np.sum(mask[:, cl]):
                    _boxes_out = np.concatenate((_boxes_out, _boxes_reshape[mask[:, cl]]), axis=0)
                    _scores_out = np.concatenate((_scores_out, _box_scores_reshape[mask[:, cl]]), axis=0)
                    _class_param_out = np.concatenate((_class_param_out, _class_param_reshape[mask[:, cl]]), axis=0)
            _boxes_out = _boxes_out[1:].astype('int')
            _scores_out = _scores_out[1:]
            _class_param_out = _class_param_out[1:]
            _conf_param = (_scores_out / _class_param_out)[:, :1]
            pick, _ = CreateArray().non_max_suppression_fast(_boxes_out, _scores_out, sensitivity)
            return np.concatenate([_boxes_out[pick], _conf_param[pick], _scores_out[pick]], axis=-1)
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def plot_boxes(true_bb, pred_bb, img_path, name_classes, colors, image_id, add_only_true=False, plot_true=True,
                   image_size=(416, 416), save_path='', return_mode='deploy'):
        method_name = 'plot_boxes'
        try:
            image = Image.open(img_path)
            # image = image.resize(image_size, Image.BICUBIC)
            # image_path = os.path.join(
            #     dataset_path, options.dataframe['val'][options.dataframe['val'].columns[0]][index])
            # img = Image.open(image_path)
            real_size = image.size
            scale_w = int(real_size[0] / image_size[0])
            scale_h = int(real_size[1] / image_size[1])

            def resize_bb(boxes, scale_width, scale_height):
                coord = boxes[:, :4]
                resized_coord = np.concatenate(
                    [coord[:, 0:1] * scale_height, coord[:, 1:2] * scale_width,
                     coord[:, 2:3] * scale_height, coord[:, 3:4] * scale_width], axis=-1)
                resized_coord = np.concatenate([resized_coord, boxes[4:]], axis=-1)
                return resized_coord

            true_bb = resize_bb(true_bb, scale_w, scale_h)
            pred_bb = resize_bb(pred_bb, scale_w, scale_h)

            def draw_box(draw, box, color, thickness, label=None, label_size=None,
                         dash_mode=False, show_label=False):
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int'))
                left = max(0, np.floor(left + 0.5).astype('int'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int'))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                if dash_mode:
                    for cur_y in [top, bottom]:
                        for x in range(left, right, 4):
                            draw.line([(x, cur_y), (x + thickness, cur_y)], fill=color, width=2)
                    for cur_y in [left, right]:
                        for x in range(top, bottom, 4):
                            draw.line([(cur_y, x), (cur_y + thickness, x)], fill=color, width=2)
                else:
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=color,
                        )

                if show_label:
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=color
                    )
                    draw.text(tuple(text_origin), label, fill=(255, 255, 255), font=font)
                return draw

            font = ImageFont.load_default()
            thickness = (image.size[0] + image.size[1]) // 300
            image_pred = image.copy()
            if plot_true:
                classes = np.argmax(true_bb[:, 5:], axis=-1)
                for i, box in enumerate(true_bb[:, :4]):
                    draw = ImageDraw.Draw(image_pred)
                    true_class = name_classes[classes[i]]
                    label = '{}'.format(true_class)
                    label_size = draw.textsize(label, font)
                    draw = draw_box(draw, box, colors[classes[i]], thickness,
                                    label=label, label_size=label_size,
                                    dash_mode=False, show_label=True)
                    del draw

            classes = np.argmax(pred_bb[:, 5:], axis=-1)
            for i, box in enumerate(pred_bb[:, :4]):
                draw = ImageDraw.Draw(image_pred)
                predicted_class = name_classes[classes[i]]
                score = pred_bb[:, 5:][i][classes[i]]  # * pred_bb[:, 4][i]
                label = '{} {:.2f}'.format(predicted_class, score)
                label_size = draw.textsize(label, font)
                draw = draw_box(draw, box, colors[classes[i]], thickness,
                                label=label, label_size=label_size,
                                dash_mode=True, show_label=True)
                del draw

            save_predict_path = os.path.join(
                save_path, f"{return_mode}_od_{'predict_true' if plot_true else 'predict'}_image_{image_id}.webp")
            image_pred.save(save_predict_path)

            save_true_path = ''
            if add_only_true:
                image_true = image.copy()
                classes = np.argmax(true_bb[:, 5:], axis=-1)
                for i, box in enumerate(true_bb[:, :4]):
                    draw = ImageDraw.Draw(image_true)
                    true_class = name_classes[classes[i]]
                    label = '{}'.format(true_class)
                    label_size = draw.textsize(label, font)
                    draw = draw_box(draw, box, colors[classes[i]], thickness,
                                    label=label, label_size=label_size,
                                    dash_mode=False, show_label=True)
                    del draw

                save_true_path = os.path.join(save_path, f"{return_mode}_od_true_image_{image_id}.webp")
                image_true.save(save_true_path)

            return save_predict_path, save_true_path
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def get_yolo_example_statistic(true_bb, pred_bb, name_classes, sensitivity=0.25):
        method_name = 'get_yolo_example_statistic'
        try:
            compat = {
                'recognize': {
                    "empty": [],
                    'unrecognize': []
                },
                'class_stat': {},
                'total_stat': {}
            }
            for name in name_classes:
                compat['recognize'][name] = []

            predict = {}
            for i, k in enumerate(pred_bb[:, :4]):
                predict[i] = {
                    'pred_class': name_classes[np.argmax(pred_bb[:, 5:][i])],
                    'conf': pred_bb[:, 4][i].item(),
                    'class_conf': pred_bb[:, 5:][i][np.argmax(pred_bb[:, 5:][i])],
                }

            count = 0
            total_conf = 0
            total_class = 0
            total_overlap = 0
            all_true = list(np.arange(len(true_bb)))
            for i, tr in enumerate(true_bb[:, :4]):
                for j, pr in enumerate(pred_bb[:, :4]):
                    boxes = np.array([true_bb[:, :4][i], pred_bb[:, :4][j]])
                    scores = np.array([true_bb[:, 5:][i], pred_bb[:, 5:][j]])
                    pick, _ = CreateArray().non_max_suppression_fast(boxes, scores, sensitivity=sensitivity)
                    if len(pick) == 1:
                        mean_iou = CreateArray().bboxes_iou(boxes[0], boxes[1])
                        compat['recognize'][name_classes[np.argmax(true_bb[:, 5:][i], axis=-1)]].append(
                            {
                                'pred_class': name_classes[np.argmax(pred_bb[:, 5:][j], axis=-1)],
                                'conf': pred_bb[:, 4][j].item(),
                                'class_conf': pred_bb[:, 5:][j][np.argmax(pred_bb[:, 5:][j], axis=-1)],
                                'class_result': True if np.argmax(true_bb[:, 5:][i], axis=-1) == np.argmax(
                                    pred_bb[:, 5:][j], axis=-1) else False,
                                'overlap': mean_iou.item()
                            }
                        )
                        if np.argmax(true_bb[:, 5:][i], axis=-1) == np.argmax(pred_bb[:, 5:][j], axis=-1):
                            count += 1
                            total_conf += pred_bb[:, 4][j].item()
                            total_class += pred_bb[:, 5:][j][np.argmax(pred_bb[:, 5:][j], axis=-1)]
                            total_overlap += mean_iou.item()

                        try:
                            predict.pop(j)
                            all_true.pop(all_true.index(i))
                        except:
                            continue

            for val in predict.values():
                compat['recognize']['empty'].append(val)

            if all_true:
                for idx in all_true:
                    compat['recognize']['unrecognize'].append(
                        {
                            "class_name": name_classes[np.argmax(true_bb[idx, 5:], axis=-1)]
                        }
                    )

            for cl in compat['recognize'].keys():
                if cl != 'empty' and cl != 'unrecognize':
                    mean_conf = 0
                    mean_class = 0
                    mean_overlap = 0
                    for pr in compat['recognize'][cl]:
                        if pr['class_result']:
                            mean_conf += pr['conf']
                            mean_class += pr['class_conf']
                            mean_overlap += pr['overlap']
                    compat['class_stat'][cl] = {
                        'mean_conf': mean_conf / len(compat['recognize'][cl]) if len(compat['recognize'][cl]) else None,
                        'mean_class': mean_class / len(compat['recognize'][cl]) if len(
                            compat['recognize'][cl]) else None,
                        'mean_overlap': mean_overlap / len(compat['recognize'][cl]) if len(
                            compat['recognize'][cl]) else None
                    }
            compat['total_stat'] = {
                'total_conf': total_conf / count if count else 0.,
                'total_class': total_class / count if count else 0.,
                'total_overlap': total_overlap / count if count else 0.,
                'total_metric': (total_conf + total_class + total_overlap) / 3 / count if count else 0.
            }
            return compat
        except Exception as e:
            print_error("CreateArray", method_name, e)

    @staticmethod
    def _round_list(x: list) -> list:
        method_name = '_round_list'
        try:
            update_x = []
            for data in x:
                if data > 1:
                    update_x.append(np.round(data, -int(math.floor(math.log10(abs(data))) - 3)).item())
                else:
                    update_x.append(np.round(data, -int(math.floor(math.log10(abs(data))) - 2)).item())
            return update_x
        except Exception as e:
            print_error("CreateArray", method_name, e)
