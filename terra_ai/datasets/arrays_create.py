import copy
import string

import matplotlib
from PIL import Image, UnidentifiedImageError
from pandas import DataFrame
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.data.training.extra import ExampleChoiceTypeChoice
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

from terra_ai.settings import DEPLOY_PRESET_PERCENT


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
        if open_tags:
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
            instructions['parameters']['classes_names'] = ["Не изменился", "Вверх", "Вниз"]
            instructions['parameters']['num_classes'] = 3

        return instructions

    @staticmethod
    def instructions_object_detection(paths_list: list, **options: dict) -> dict:

        instructions = {'instructions': paths_list,
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
                                       'cols_names': options['cols_names']
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
            frames_count = int(cap.get(7))
            frames_number = 0
            save_path = os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                     f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
            instructions_paths.append(save_path)
            output_movie = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), orig_shape)
            stop_flag = False
            while not stop_flag:
                ret, frame = cap.read()
                frames_number += 1
                output_movie.write(frame)
                if options['video_mode'] == 'completely' and options['max_frames'] > frames_count and ret or \
                        options['video_mode'] == 'length_and_step' and options['length'] > frames_count and ret:
                    tmp_array.append(frame)
                if not ret or frames_number > frames_count:
                    stop_flag = True
            if options['video_mode'] == 'completely' and options['max_frames'] > frames_count or \
                    options['video_mode'] == 'length_and_step' and options['length'] > frames_count:
                fr_to_add, tot_frames = 0, 0
                if options['video_mode'] == 'completely':
                    fr_to_add = options['max_frames'] - frames_count
                    tot_frames = options['max_frames']
                elif options['video_mode'] == 'length_and_step':
                    fr_to_add = options['length'] - frames_count
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
                                       "num_classes": options['classes_names'],
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
                                       'classes_names': options['open_tags'],
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
    def cut_object_detection(paths_list: list, dataset_folder=None, **options: dict) -> dict:

        for elem in paths_list:
            os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
            shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                               os.path.basename(elem)))

        paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                      for elem in paths_list]

        instructions = {'instructions': paths_list,
                        'parameters': {'yolo': options['yolo'],
                                       'num_classes': options['num_classes'],
                                       'classes_names': options['classes_names'],
                                       'put': options['put']}
                        }

        return instructions

    @staticmethod
    def create_image(image_path: str, **options) -> dict:

        img = Image.open(image_path)
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

        instructions = {'instructions': np.array(row),
                        'parameters': options}

        return instructions

    @staticmethod
    def create_object_detection(annot_path: str, **options):

        """
        Args:
            annot_path: str
                Путь к файлу
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

        with open(annot_path, 'r') as coordinates:
            coords = coordinates.read()
        real_boxes = []
        for coord in coords.split('\n'):
            if coord:
                real_boxes.append([literal_eval(num) for num in coord.split(',')])

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
    def preprocess_timeseries(row: np.ndarray, **options) -> np.ndarray:

        if options["trend"]:
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
            array = np.array(array)
        else:
            array = row
            if options['scaler'] != 'no_scaler':
                orig_shape = row.shape
                array = options['preprocess'].transform(row.reshape(-1, 1))
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
        if not options.data.use_generator:
            y_true = options.Y.get('val').get(f"{output_id}")
        else:
            y_true = []
            for _, y_val in options.dataset['val'].batch(1):
                y_true.extend(y_val.get(f'{output_id}').numpy())
            y_true = np.array(y_true)
        return y_true

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
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
                inverse_x = np.zeros_like(x_val.get(input)[:, 0:1, :])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {
                            int(input): {
                                column: x_val.get(input)[:, i:i + 1, :]
                            }
                        }
                        inverse_col = options.preprocessing.inverse_data(_options).get(int(input)).get(column)
                    else:
                        inverse_col = x_val.get(input)[:, i:i + 1, :]
                    inverse_x = np.concatenate([inverse_x, inverse_col], axis=1)
                inverse_x_val[input] = inverse_x[:, 1:, :]
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_results(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        x_array, inverse_x_array = CreateArray().get_x_array(options)
        return_data = {}
        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = CreateArray().get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            # print('true_array.shape, postprocess_array.shape', true_array.shape, postprocess_array.shape)
            example_idx = CreateArray().prepare_example_idx_to_show(
                array=array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
            # print(len(example_idx), len(array), example_idx)
            if options.data.outputs[output_id].task == LayerOutputTypeChoice.Classification or \
                    options.data.outputs[output_id].task == LayerOutputTypeChoice.TimeseriesTrend:
                # y_true = CreateArray().get_y_true(options, output_id)
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

            elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Segmentation:
                return_data[output_id] = []
                data = []
                for j, cls in enumerate(options.data.outputs.get(output_id).classes_names):
                    data.append((cls, options.data.outputs.get(output_id).classes_colors[j].as_rgb_tuple()))
                for idx in example_idx:
                    input_id = list(options.data.inputs.keys())[0]
                    colors = [color.as_rgb_tuple() for color in options.data.outputs.get(output_id).classes_colors]
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
                                inverse_x_array=None if not inverse_x_array else inverse_x_array.get(f"{input_id}"),
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
                for idx in example_idx:
                    input_id = list(options.data.inputs.keys())[0]
                    source = CreateArray().postprocess_initial_source(
                        options=options,
                        input_id=input_id,
                        save_id=idx,
                        example_id=idx,
                        dataset_path=dataset_path,
                        preset_path=save_path,
                        x_array=None if not x_array else x_array.get(f"{input_id}"),
                        inverse_x_array=None if not inverse_x_array else inverse_x_array.get(f"{input_id}"),
                        return_mode='deploy'
                    )
                    actual_value, predict_values = CreateArray().postprocess_time_series()
                    return_data[output_id].append(
                        {
                            "source": source,
                            "actual": actual_value,
                            "data": predict_values
                        }
                    )

            elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Regression:
                return_data[output_id] = {
                    'preset': [],
                    'label': []
                }
                source_col = []
                for inp in options.data.inputs.keys():
                    source_col.extend(list(options.data.columns.get(inp).keys()))
                preprocess = options.preprocessing.preprocessing.get(list(options.data.outputs.keys())[0])
                for idx in example_idx:
                    row_list = []
                    for inp_col in source_col:
                        row_list.append(options.dataframe.get('val')[inp_col][idx])
                    return_data[output_id]['preset'].append(row_list)
                    for i, col in enumerate(list(options.data.columns.get(output_id).keys())):
                        if type(preprocess.get(col)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(output_id): {col: array[idx, i:i + 1].reshape(-1, 1)}}
                            inverse_col = options.preprocessing.inverse_data(_options).get(output_id).get(col)
                            inverse_col = inverse_col.squeeze().tolist()
                        else:
                            inverse_col = array[idx, i:i + 1].tolist()
                    return_data[output_id]['label'].append(inverse_col)

            else:
                return_data[output_id] = []
        return return_data

    @staticmethod
    def postprocess_initial_source(
            options,
            input_id: int,
            example_id: int,
            dataset_path: str,
            preset_path: str,
            save_id: int = None,
            x_array=None,
            inverse_x_array=None,
            return_mode='deploy'
    ):
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
            source = os.path.join(preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webp")
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
                if return_mode == 'deploy':
                    # TODO: для предикта по одному каналу
                    output_channel = list(options.data.columns.get(list(options.data.columns.keys())[0]).keys())[0]
                    for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                        if output_channel == channel:
                            source = inverse_x_array[example_id][i].astype('float').tolist()
                            break
                if return_mode == 'callback':
                    for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                        multi = True if i > 0 else False
                        names += f"«{channel.split('_', 1)[-1]}», "
                        graphics_data.append(
                            {
                                'id': i + 1,
                                'graph_name': f"График канала «{channel.split('_', 1)[-1]}»",
                                'x_label': 'Время',
                                'y_label': 'Значение',
                                'plot_data': {
                                    'x': np.arange(inverse_x_array[example_id].shape[-1]).astype('int').tolist(),
                                    'y': inverse_x_array[example_id][i].astype('float').tolist()
                                },
                            }
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
                for inp in options.data.inputs.keys():
                    for col_name in options.data.columns.get(inp).keys():
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

    @staticmethod
    def dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
        axis = tuple(np.arange(1, len(y_true.shape))) if batch_mode else None
        intersection = np.sum(y_true * y_pred, axis=axis)
        union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
        return (2.0 * intersection + smooth) / (union + smooth)

    @staticmethod
    def sort_dict(dict_to_sort: dict, mode='by_name'):
        if mode == 'by_name':
            sorted_keys = sorted(dict_to_sort)
            sorted_values = []
            for w in sorted_keys:
                sorted_values.append(dict_to_sort[w])
            return tuple(sorted_keys), tuple(sorted_values)
        elif mode == 'ascending':
            sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get)
            sorted_values = []
            for w in sorted_keys:
                sorted_values.append(dict_to_sort[w])
            return tuple(sorted_keys), tuple(sorted_values)
        elif mode == 'descending':
            sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)
            sorted_values = []
            for w in sorted_keys:
                sorted_values.append(dict_to_sort[w])
            return tuple(sorted_keys), tuple(sorted_values)
        else:
            return tuple(dict_to_sort.keys()), tuple(dict_to_sort.values())

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: str = "best", seed_idx: list = None) -> dict:
        example_idx = {}
        # out = f"{self.interactive_config.intermediate_result.main_output}"
        encoding = options.data.outputs.get(output).encoding
        # count = self.interactive_config.intermediate_result.num_examples
        # choice_type = self.interactive_config.intermediate_result.example_choice_type
        task = options.data.outputs.get(output).task
        # y_true = options.Y.get("val").get(f"{output}")
        if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                # y_pred = self.y_pred.get(out)
                if array.shape[-1] == true_array.shape[-1] and encoding == LayerEncodingChoice.ohe and true_array.shape[-1] > 1:
                    classes = np.argmax(true_array, axis=-1)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] > 1:
                    classes = copy.deepcopy(true_array)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] == 1:
                    classes = copy.deepcopy(true_array)
                else:
                    classes = copy.deepcopy(true_array)
                probs = np.array([pred[classes[i]] for i, pred in enumerate(array)])
                sorted_args = np.argsort(probs)
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx = sorted_args[::-1][:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx = sorted_args[:count]

            elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
                if encoding == LayerEncodingChoice.ohe:
                    array = to_categorical(
                        np.argmax(array, axis=-1),
                        num_classes=options.data.outputs.get(output).num_classes
                    )
                if encoding == LayerEncodingChoice.multi:
                    array = np.where(array >= 0.9, 1, 0)
                dice_val = CreateArray().dice_coef(true_array, array, batch_mode=True)
                dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = CreateArray().sort_dict(dice_dict, mode="descending")
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = CreateArray().sort_dict(dice_dict, mode="ascending")
                    example_idx = example_idx[:count]

            elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                delta = np.abs(true_array - array) * 100 / true_array
                while len(delta.shape) != 1:
                    delta = np.mean(delta, axis=-1)
                delta_dict = dict(zip(np.arange(0, len(delta)), delta))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = CreateArray().sort_dict(delta_dict, mode="ascending")
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = CreateArray().sort_dict(delta_dict, mode="descending")
                    example_idx = example_idx[:count]
            else:
                pass

        elif choice_type == ExampleChoiceTypeChoice.seed and seed_idx:
            example_idx = seed_idx[:count]

        elif choice_type == ExampleChoiceTypeChoice.random:
            example_idx = np.random.randint(0, len(true_array), count)
        else:
            pass
        return example_idx

    @staticmethod
    def postprocess_classification(
            predict_array: np.ndarray,
            true_array: np.ndarray,
            options: DatasetOutputsData,
            show_stat: bool = False,
            return_mode='deploy'
    ):
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

    @staticmethod
    def postprocess_segmentation(
            predict_array: np.ndarray,
            true_array: np.ndarray,
            options: DatasetOutputsData,
            output_id: int,
            image_id: int,
            save_path: str,
            colors: list = None,
            return_mode='deploy',
            show_stat: bool = False
    ):
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

    @staticmethod
    def postprocess_text_segmentation(
            pred_array: np.ndarray,
            options: DatasetOutputsData,
            dataframe: DataFrame,
            dataset_params: dict,
            example_id: int,
            return_mode='deploy',
            class_colors: list = None,
            show_stat: bool = False,
            true_array: np.ndarray = None,
    ):

        def add_tags_to_word(word: str, tag: str):
            if tag:
                for t in tag:
                    word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
                return word
            else:
                return f"<p1>{word}</p1>"

        # def color_mixer(colors: list):
        #     if colors:
        #         result = np.zeros((3,))
        #         for color in colors:
        #             result += np.array(color)
        #         result = result / len(colors)
        #         return tuple(result.astype('int').tolist())
        #
        # def tag_mixer(tags: list, colors: dict):
        #     tags = sorted(tags, reverse=False)
        #     mix_tag = f"{tags[0][1:-1]}"
        #     for tag in tags[1:]:
        #         mix_tag += f"+{tag[1:-1]}"
        #     return f"<{mix_tag}>", color_mixer([colors[tag] for tag in tags])
        #
        def reformat_tags(y_array: np.ndarray, tag_list: list,  # classes_names: dict, colors: dict,
                          sensitivity: float = 0.9):
            norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
            reformat_tags = []
            for word_tag in norm_array:
                if np.sum(word_tag) == 0:
                    reformat_tags.append(None)
                # elif np.sum(word_tag) == 1:
                #     reformat_tags.append([tag_list[np.argmax(word_tag, axis=-1)]])
                else:
                    mix_tag = []
                    # mix_name = ""
                    for i, tag in enumerate(word_tag):
                        if tag == 1:
                            mix_tag.append(tag_list[i])
                            # mix_name += f"{classes_names[tag_list[i]]} + "
                    # mix_tag, mix_color = tag_mixer(mix_tag, colors)
                    # if mix_tag not in classes_names.keys():
                    #     classes_names[mix_tag] = mix_name[:-3]
                    #     colors[mix_tag] = mix_color
                    reformat_tags.append(mix_tag)
            return reformat_tags

        def text_colorization(text: str, label_array: np.ndarray, tag_list: list, class_names: dict, colors: dict):
            text = text.split(" ")
            # norm_array = np.where(label_array >= sensitivity, 1, 0).astype('int')
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
            for i, name in enumerate(dataset_tags):
                colors[name] = tuple(np.random.randint(256, size=3).tolist())
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
            data = []
            for tag in colors.keys():
                data.append(
                    (tag, classes_names[tag], colors[tag])
                )
            return initinal_text, text_segmentation, data

        if return_mode == 'callback':
            data = {
                "y_true": {},
                "y_pred": {},
                "tags_color": {},
                "stat": {}
            }
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
                "data": [
                    {
                        "title": "Текст",
                        "value": true_text_segmentation,
                        "color_mark": None
                    }
                ]
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
                "data": [
                    {
                        "title": "Текст",
                        "value": pred_text_segmentation,
                        "color_mark": None
                    }
                ]
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
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': "-",
                                'color_mark': None
                            }
                        )
                    else:
                        dice_val = np.round(
                            CreateArray().dice_coef(y_true[:, idx], y_pred[:, idx], batch_mode=False) * 100, 1)
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': f"{dice_val} %",
                                'color_mark': 'success' if dice_val >= 90 else 'wrong'
                            }
                        )
                        count += 1
                        mean_val += dice_val
                if count and mean_val / count >= 90:
                    mean_color_mark = "success"
                elif count and mean_val / count < 90:
                    mean_color_mark = "wrong"
                else:
                    mean_color_mark = None
                data["stat"]["data"].insert(
                    0,
                    {
                        'title': "Средняя точность",
                        'value': f"{round(mean_val / count, 2)}%" if count else "-",
                        'color_mark': mean_color_mark
                    }
                )
            return data

    @staticmethod
    def postprocess_regression(
            column_names: list,
            inverse_y_true: np.ndarray,
            inverse_y_pred: np.ndarray,
            show_stat: bool = False,
            return_mode='deploy',
    ):
        data = {"y_true": {
            "type": "str",
            "data": []
        }}
        if return_mode == 'deploy':
            source = []
            return source
        else:
            for i, name in enumerate(column_names):
                data["y_true"]["data"].append(
                    {
                        "title": name.split('_', 1)[-1],
                        "value": f"{inverse_y_true[i]: .2f}",
                        "color_mark": None
                    }
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
                data["stat"] = {
                    "type": "str",
                    "data": []
                }
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

    @staticmethod
    def postprocess_time_series(
            options: DatasetData,
            real_x: np.ndarray,
            inverse_y_true: np.ndarray,
            inverse_y_pred: np.ndarray,
            output_id: int,
            depth: int,
            show_stat: bool = False,
            templates: list = None
    ):
        """
        real_x = self.inverse_x_val.get(f"{input}")[example_idx]
        inverse_y_true = self.inverse_y_true.get("val").get(output_id)[example_idx]
        inverse_y_pred = self.inverse_y_pred.get(output_id)[example_idx]
        depth = self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]
        templates = [self._fill_graph_plot_data, self._fill_graph_front_structure]
        """
        data = {
            "y_true": {},
            "y_pred": {},
            "stat": {}
        }
        graphics = []
        _id = 1
        for i, channel in enumerate(options.columns.get(output_id).keys()):
            for inp in options.inputs.keys():
                for input_column in options.columns.get(inp).keys():
                    if channel.split("_", 1)[-1] == input_column.split("_", 1)[-1]:
                        init_column = list(options.columns.get(inp).keys()).index(input_column)
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
                                        x=np.arange(real_x.shape[-1]).astype('int').tolist(),
                                        y=np.array(real_x[init_column]).astype('float').tolist()
                                    ),
                                    templates[0](
                                        label="Истинное значение",
                                        x=np.arange(real_x.shape[-1], real_x.shape[-1] + depth).astype('int').tolist(),
                                        y=inverse_y_true[i].astype('float').tolist()
                                    ),
                                    templates[0](
                                        label="Предсказанное значение",
                                        x=np.arange(real_x.shape[-1], real_x.shape[-1] + depth).astype('int').tolist(),
                                        y=inverse_y_pred[i].astype('float').tolist()
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
                for step in range(inverse_y_true.shape[-1]):
                    deviation = (inverse_y_pred[i, step] - inverse_y_true[i, step]) * 100 / inverse_y_true[i, step]
                    data["stat"]["data"][-1]["value"].append(
                        {
                            "Шаг": f"{step + 1}",
                            "Истина": f"{inverse_y_true[i, step].astype('float'): .2f}",
                            "Предсказание": f"{inverse_y_pred[i, step].astype('float'): .2f}",
                            "Отклонение": {
                                "value": f"{deviation: .2f} %",
                                "color_mark": "success" if abs(deviation) < 2 else "wrong"
                            }
                        }
                    )
        return data

    @staticmethod
    def postprocess_object_detection():
        pass
