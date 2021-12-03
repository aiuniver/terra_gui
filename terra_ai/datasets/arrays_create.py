from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    DataframeClassificationCallback, AudioClassificationCallback, VideoClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.datasets.utils import get_yolo_anchors, resize_bboxes, Yolo_terra, Voc, Coco, Udacity, Kitti, Yolov1
from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerScalerVideoChoice, LayerPrepareMethodChoice
from terra_ai.data.datasets.extra import LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice, \
    LayerTextModeChoice, LayerAudioModeChoice, LayerVideoModeChoice, LayerScalerAudioChoice
from terra_ai.data.datasets.creations.layers.output.types.ObjectDetection import LayerODDatasetTypeChoice


import os
import re
import cv2
import numpy as np
import pandas as pd
import shutil
import pymorphy2
import random
import librosa.feature as librosa_feature
import imgaug
from PIL import UnidentifiedImageError
from ast import literal_eval
from sklearn.cluster import KMeans
from pydub import AudioSegment
from librosa import load as librosa_load
from pydantic.color import Color
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import utils
from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum


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
        annot_type = options['model_type']
        if annot_type == LayerODDatasetTypeChoice.Yolo_terra:
            for path in paths_list:
                with open(path, 'r') as coordinates:
                    coordinate = coordinates.read()
                coordinates_list.append(' '.join([coord for coord in coordinate.split('\n') if coord]))

        else:
            model_type = eval(f'{annot_type}()')
            data, cls_hierarchy = model_type.parse(paths_list, options['classes_names'])
            yolo_terra = Yolo_terra(options['classes_names'], cls_hierarchy=cls_hierarchy)
            data = yolo_terra.generate(data)
            for key in data:
                coordinates_list.append(data[key])

        instructions = {'instructions': coordinates_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_image(paths_list: list, dataset_folder=None, **options: dict):

        # for elem in paths_list:
        #     os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
        #     shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
        #                                        os.path.basename(elem)))

        # paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                      # for elem in paths_list]

        image_mode = 'stretch' if not options.get('image_mode') else options['image_mode']

        instructions = {'instructions': paths_list,
                        'parameters': {'height': options['height'],
                                       'width': options['width'],
                                       'net': options['net'],
                                       'image_mode': image_mode,
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
                        'parameters': {'classes_names': options['classes_names'],
                                       'encoding': 'ohe',
                                       'num_classes': options['num_classes'],
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

        # for elem in paths_list:
        #     os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
        #     shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
        #                                        os.path.basename(elem)))

        # paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
        #               for elem in paths_list]

        instructions = {'instructions': paths_list,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': options['num_classes'],
                                       'height': options['height'],
                                       'width': options['width'],
                                       'classes_colors': options['classes_colors'],
                                       'classes_names': options['classes_names'],
                                       'cols_names': options['cols_names'],
                                       'put': options['put'],
                                       'encoding': 'ohe'
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
                                       'cols_names': options['cols_names'],
                                       'put': options['put'],
                                       'num_classes': options['num_classes'],
                                       'classes_names': options['classes_names'],
                                       'length': options['length'],
                                       'encoding': 'multi'
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
                                       'put': options['put'],
                                       'frame_mode': options['frame_mode']}
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

        if coords:
            real_boxes = resize_bboxes(options['frame_mode'], coords, options['orig_x'], options['orig_y'])
        else:
            real_boxes = [[0, 0, 0, 0, 0]]

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
    def preprocess_image(array: np.ndarray, **options) -> tuple:

        def augmentation_image(image_array, coords, augmentation_dict):

            # КОСТЫЛЬ ИЗ-ЗА .NATIVE()
            for key, value in augmentation_dict.items():
                if value:
                    for name, elem in value.items():
                        if key != 'ChannelShuffle':
                            if isinstance(augmentation_dict[key][name], list):
                                augmentation_dict[key][name] = tuple(augmentation_dict[key][name])
                            elif isinstance(augmentation_dict[key][name], dict):
                                for name2, elem2 in augmentation_dict[key][name].items():
                                    augmentation_dict[key][name][name2] = tuple(augmentation_dict[key][name][name2])

            aug_parameters = []
            for key, value in augmentation_dict.items():
                if value:
                    aug_parameters.append(getattr(imgaug.augmenters, key)(**value))
            augmentation_object = imgaug.augmenters.Sequential(aug_parameters, random_order=True)

            augmentation_object_data = {'images': np.expand_dims(image_array, axis=0)}
            if coords:
                coords = coords.split(' ')
                for i in range(len(coords)):
                    coords[i] = [float(x) for x in coords[i].split(',')]
                bounding_boxes = []
                for elem in coords:
                    bounding_boxes.append(imgaug.BoundingBox(*elem))
                bounding_boxes = imgaug.augmentables.bbs.BoundingBoxesOnImage(bounding_boxes,
                                                                              shape=(image_array.shape[0],
                                                                                     image_array.shape[1])
                                                                              )
                augmentation_object_data.update([('bounding_boxes', bounding_boxes)])

            image_array_aug = augmentation_object(**augmentation_object_data)
            if coords:
                bounding_boxes_aug = image_array_aug[1]
                image_array_aug = image_array_aug[0][0]
                bounding_boxes_aug = bounding_boxes_aug.remove_out_of_image().clip_out_of_image()
                aug_coords = []
                for elem in bounding_boxes_aug.bounding_boxes:
                    aug_coords.append(
                        ','.join(str(x) for x in [elem.x1_int, elem.y1_int, elem.x2_int, elem.y2_int, elem.label]))
                aug_coords = ' '.join(aug_coords)
                if not aug_coords:
                    aug_coords = ''
                return image_array_aug, aug_coords
            else:
                return image_array_aug

        def resize_frame(image_array, target_shape, frame_mode):

            original_shape = (image_array.shape[0], image_array.shape[1])
            resized = None
            if frame_mode == 'stretch':
                resized = cv2.resize(image_array, (target_shape[1], target_shape[0]))
            elif frame_mode == 'fit':
                if image_array.shape[1] >= image_array.shape[0]:
                    resized_shape = list(target_shape).copy()
                    resized_shape[0] = int(image_array.shape[0] / (image_array.shape[1] / target_shape[1]))
                    if resized_shape[0] > target_shape[0]:
                        resized_shape = list(target_shape).copy()
                        resized_shape[1] = int(image_array.shape[1] / (image_array.shape[0] / target_shape[0]))
                    image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
                elif image_array.shape[0] >= image_array.shape[1]:
                    resized_shape = list(target_shape).copy()
                    resized_shape[1] = int(image_array.shape[1] / (image_array.shape[0] / target_shape[0]))
                    if resized_shape[1] > target_shape[1]:
                        resized_shape = list(target_shape).copy()
                        resized_shape[0] = int(image_array.shape[0] / (image_array.shape[1] / target_shape[1]))
                    image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
                resized = image_array
                if resized.shape[0] < target_shape[0]:
                    black_bar = np.zeros((int((target_shape[0] - resized.shape[0]) / 2), resized.shape[1], 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    black_bar_2 = np.zeros((int((target_shape[0] - resized.shape[0])), resized.shape[1], 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2))
                if resized.shape[1] < target_shape[1]:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1]) / 2), 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    black_bar_2 = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2), axis=1)
            elif frame_mode == 'cut':
                resized = image_array.copy()
                if original_shape[0] > target_shape[0]:
                    resized = resized[int(original_shape[0] / 2 - target_shape[0] / 2):int(
                        original_shape[0] / 2 - target_shape[0] / 2) + target_shape[0], :]
                else:
                    black_bar = np.zeros((int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    black_bar_2 = np.zeros((int((target_shape[0] - resized.shape[0])), original_shape[1], 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2))
                if original_shape[1] > target_shape[1]:
                    resized = resized[:, int(original_shape[1] / 2 - target_shape[1] / 2):int(
                        original_shape[1] / 2 - target_shape[1] / 2) + target_shape[1]]
                else:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    black_bar_2 = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2), axis=1)
            return resized

        augm_data = None
        if options.get('augmentation') and options.get('augm_data'):
            array, augm_data = augmentation_image(image_array=array,
                                                  coords=options['augm_data'],
                                                  augmentation_dict=options['augmentation'])

        frame_mode = options['image_mode'] if 'image_mode' in options.keys() else 'stretch' # Временное решение

        array = resize_frame(image_array=array,
                             target_shape=(options['height'], options['width']),
                             frame_mode=frame_mode)

        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            if options['scaler'] == 'min_max_scaler':
                orig_shape = array.shape
                array = options['preprocess'].transform(array.reshape(-1, 1))
                array = array.reshape(orig_shape).astype('float32')
            elif options['scaler'] == 'terra_image_scaler':
                array = options['preprocess'].transform(array)

        if isinstance(augm_data, str):
            return array, augm_data
        else:
            return array

    @staticmethod
    def preprocess_video(array: np.ndarray, **options) -> np.ndarray:

        def resize_frame(image_array, target_shape, frame_mode):

            original_shape = (image_array.shape[0], image_array.shape[1])
            resized = None
            if frame_mode == 'stretch':
                resized = cv2.resize(image_array, (target_shape[1], target_shape[0]))
            elif frame_mode == 'fit':
                if image_array.shape[1] >= image_array.shape[0]:
                    resized_shape = list(target_shape).copy()
                    resized_shape[0] = int(image_array.shape[0] / (image_array.shape[1] / target_shape[1]))
                    if resized_shape[0] > target_shape[0]:
                        resized_shape = list(target_shape).copy()
                        resized_shape[1] = int(image_array.shape[1] / (image_array.shape[0] / target_shape[0]))
                    image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
                elif image_array.shape[0] >= image_array.shape[1]:
                    resized_shape = list(target_shape).copy()
                    resized_shape[1] = int(image_array.shape[1] / (image_array.shape[0] / target_shape[0]))
                    if resized_shape[1] > target_shape[1]:
                        resized_shape = list(target_shape).copy()
                        resized_shape[0] = int(image_array.shape[0] / (image_array.shape[1] / target_shape[1]))
                    image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
                resized = image_array
                if resized.shape[0] < target_shape[0]:
                    black_bar = np.zeros((int((target_shape[0] - resized.shape[0]) / 2), resized.shape[1], 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    black_bar_2 = np.zeros((int((target_shape[0] - resized.shape[0])), resized.shape[1], 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2))
                if resized.shape[1] < target_shape[1]:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1]) / 2), 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    black_bar_2 = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2), axis=1)
            elif frame_mode == 'cut':
                resized = image_array.copy()
                if original_shape[0] > target_shape[0]:
                    resized = resized[int(original_shape[0] / 2 - target_shape[0] / 2):int(
                        original_shape[0] / 2 - target_shape[0] / 2) + target_shape[0], :]
                else:
                    black_bar = np.zeros((int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized))
                    black_bar_2 = np.zeros((int((target_shape[0] - resized.shape[0])), original_shape[1], 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2))
                if original_shape[1] > target_shape[1]:
                    resized = resized[:, int(original_shape[1] / 2 - target_shape[1] / 2):int(
                        original_shape[1] / 2 - target_shape[1] / 2) + target_shape[1]]
                else:
                    black_bar = np.zeros((target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3),
                                         dtype='uint8')
                    resized = np.concatenate((black_bar, resized), axis=1)
                    black_bar_2 = np.zeros((target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                                           dtype='uint8')
                    resized = np.concatenate((resized, black_bar_2), axis=1)
            return resized

        trgt_shape = (options['height'], options['width'])
        resized_array = []
        for i in range(len(array)):
            if array[i].shape[1:-1] != trgt_shape:
                resized_array.append(resize_frame(image_array=array[i],
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
    def postprocess_results(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                            threashold=0.1) -> dict:
        print('postprocess_results', options.data.architecture)
        return_data = {}
        if options.data.architecture == ArchitectureChoice.ImageClassification:
            return_data = ImageClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TextClassification:
            return_data = TextClassificationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.DataframeClassification:
            return_data = DataframeClassificationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.AudioClassification:
            return_data = AudioClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.VideoClassification:
            return_data = VideoClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TimeseriesTrend:
            return_data = TimeseriesTrendCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.ImageSegmentation:
            return_data = ImageSegmentationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TextSegmentation:
            return_data = TextSegmentationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.DataframeRegression:
            # print('options.data.architecture == ArchitectureChoice.DataframeRegression')
            return_data = DataframeRegressionCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.Timeseries:
            return_data = TimeseriesCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.YoloV3:
            return_data = YoloV3Callback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path,
                sensitivity=sensitivity, threashold=threashold
            )
        elif options.data.architecture == ArchitectureChoice.YoloV4:
            return_data = YoloV4Callback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path,
                sensitivity=sensitivity, threashold=threashold
            )
        else:
            pass
        return return_data

