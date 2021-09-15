from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerScalerVideoChoice, LayerPrepareMethodChoice
from terra_ai.data.datasets.extra import LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice,\
    LayerTextModeChoice, LayerAudioModeChoice, LayerVideoModeChoice, LayerScalerAudioChoice

import os
import re
import cv2
import numpy as np
import pandas as pd
import shutil
import pymorphy2
import librosa.feature as librosa_feature
from ast import literal_eval
from sklearn.cluster import KMeans
from pydub import AudioSegment
from librosa import load as librosa_load
from pydantic.color import Color
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import utils


class CreateArray(object):

    @staticmethod
    def instructions_image(paths_list: list, **options: dict) -> dict:

        instructions = {'instructions': paths_list,
                        'parameters': {'height': options['parameters']['height'],
                                       'width': options['parameters']['width'],
                                       'net': options['parameters']['net'],
                                       'object_detection': options['parameters']['object_detection'],
                                       'scaler': options['parameters']['scaler'],
                                       'max_scaler': options['parameters']['max_scaler'],
                                       'min_scaler': options['parameters']['min_scaler'],
                                       'put': options['id']}}

        return instructions

    @staticmethod
    def instructions_video(paths_list: list, **options: dict) -> dict:

        video: list = []
        cur_step = 0

        for elem in paths_list:
            if options['parameters']['video_mode'] == LayerVideoModeChoice.completely:
                video.append(';'.join([elem, f'[{cur_step}-{options["parameters"]["max_frames"]}]']))
            elif options['parameters']['video_mode'] == LayerVideoModeChoice.length_and_step:
                cur_step = 0
                stop_flag = False
                cap = cv2.VideoCapture(elem)
                frame_count = int(cap.get(7))
                while not stop_flag:
                    video.append(';'.join([elem, f'[{cur_step}-{cur_step + options["parameters"]["length"]}]']))
                    cur_step += options['parameters']['step']
                    if cur_step + options['parameters']['length'] > frame_count:
                        stop_flag = True
                        if options['parameters']['length'] < frame_count:
                            video.append(
                                ';'.join([elem, f'[{frame_count - options["parameters"]["length"]}-{frame_count}]']))

        instructions = {'instructions': video,
                        'parameters': {'height': options['parameters']['height'],
                                       'width': options['parameters']['width'],
                                       'put': options['id'],
                                       'min_scaler': options['parameters']['min_scaler'],
                                       'max_scaler': options['parameters']['max_scaler'],
                                       'scaler': options['parameters']['scaler'],
                                       'frame_mode': options['parameters']['frame_mode'],
                                       'fill_mode': options['parameters']['fill_mode'],
                                       'video_mode': options['parameters']['video_mode'],
                                       'length': options['parameters']['length'],
                                       'max_frames': options['parameters']['max_frames']}}

        return instructions

    @staticmethod
    def instructions_audio(paths_list: list, **options: dict) -> dict:

        audio: list = []

        for elem in paths_list:
            if options['parameters']['audio_mode'] == LayerAudioModeChoice.completely:
                audio.append(';'.join([elem, f'[0.0-{options["parameters"]["max_seconds"]}]']))
            elif options['parameters']['audio_mode'] == LayerAudioModeChoice.length_and_step:
                cur_step = 0.0
                stop_flag = False
                sample_length = AudioSegment.from_file(elem).duration_seconds
                while not stop_flag:
                    audio.append(';'.join([elem, f'[{cur_step}-{cur_step + options["parameters"]["max_seconds"]}]']))
                    cur_step += options['parameters']['step']
                    cur_step = round(cur_step, 1)
                    if cur_step + options['parameters']['length'] > sample_length:
                        stop_flag = True

        instructions = {'instructions': audio,
                        'parameters': {'sample_rate': options['parameters']['sample_rate'],
                                       'parameter': options['parameters']['parameter'],
                                       'scaler': options['parameters']['scaler'],
                                       'max_scaler': options['parameters']['max_scaler'],
                                       'min_scaler': options['parameters']['min_scaler'],
                                       'put': options['id']}}

        return instructions

    @staticmethod
    def instructions_text(paths_list: list, **options: dict) -> dict:

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

        txt_list: dict = {}
        text: dict = {}
        lower: bool = True
        open_tags, close_tags = None, None
        open_symbol, close_symbol = None, None
        if options['parameters'].get('open_tags'):
            open_tags, close_tags = options['parameters']['open_tags'].split(' '), options['parameters'][
                'close_tags'].split(' ')
        if open_tags:
            open_symbol = open_tags[0][0]
            close_symbol = close_tags[0][-1]
        length = options['parameters']['length'] if options['parameters'][
                                                        'text_mode'] == LayerTextModeChoice.length_and_step else \
        options['parameters']['max_words']

        for path in paths_list:
            text_file = read_text(file_path=path, lower=lower, del_symbols=options['parameters']['filters'], split=' ',
                                  open_symbol=open_symbol, close_symbol=close_symbol)
            if text_file:
                txt_list[path] = text_file
        ### ДОБАВИТЬ ОТКРЫТИЕ ИЗ ТАБЛИЦЫ
        if open_symbol:
            for key in txt_list.keys():
                words = []
                for word in txt_list[key].split(' '):
                    if not word in open_tags + close_tags:
                        words.append(word)
                txt_list[key] = ' '.join(words)

        if options['parameters']['pymorphy']:
            pymorphy = pymorphy2.MorphAnalyzer()
            for key, value in txt_list.items():
                txt_list[key] = apply_pymorphy(value, pymorphy)

        for key, value in sorted(txt_list.items()):
            if options['parameters']['text_mode'] == LayerTextModeChoice.completely:
                text[';'.join([key, f'[0-{options["parameters"]["max_words"]}]'])] = ' '.join(
                    value.split(' ')[:options['parameters']['max_words']])
            elif options['parameters']['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(value.split(' '))
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text[';'.join([key, f'[{cur_step}-{cur_step + length}]'])] = ' '.join(
                        value.split(' ')[cur_step: cur_step + length])
                    cur_step += options['parameters']['step']
                    if cur_step + options['parameters']['length'] > max_length:
                        stop_flag = True

        instructions = {'instructions': text,
                        'parameters': {'prepare_method': options['parameters']['prepare_method'],
                                       'put': options['id'],
                                       'length': length,
                                       'max_words_count': options['parameters']['max_words_count'],
                                       'word_to_vec_size': options['parameters'].get('word_to_vec_size'),
                                       'filters': options['parameters']['filters']
                                       },
                        }

        return instructions

    @staticmethod
    def instructions_classification(paths_list: list, **options: dict) -> dict:

        if os.path.isfile(options['parameters']['sources_paths'][0]) and\
                options['parameters']['sources_paths'][0].endswith('.csv'):
            file_name = options['parameters']['sources_paths'][0]
            data = pd.read_csv(file_name, usecols=options['parameters']['cols_names'])
            column = data[options['parameters']['cols_names'][0]].to_list()
            classes_names = []
            for elem in column:
                if elem not in classes_names:
                    classes_names.append(elem)
            num_classes = len(classes_names)
        else:
            classes_names = sorted([os.path.basename(elem) for elem in options['parameters']['sources_paths']])
            num_classes = len(classes_names)

        instructions = {'instructions': paths_list,
                        'parameters': {"one_hot_encoding": options['parameters']['one_hot_encoding'],
                                       "classes_names": classes_names,
                                       "num_classes": num_classes,
                                       'put': options['id']
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_regression(number_list: list, **options: dict) -> dict:

        options["put"] = options['id']

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def instructions_segmentation(paths_list: list, **options: dict) -> dict:

        instructions = {'instructions': paths_list,
                        'parameters': {'mask_range': options['parameters']['mask_range'],
                                       'num_classes': len(options['parameters']['classes_names']),
                                       'height': options['parameters']['height'],
                                       'width': options['parameters']['width'],
                                       'classes_colors': [Color(color).as_rgb_tuple() for color in
                                                          options['parameters']['classes_colors']],
                                       'classes_names': options['parameters']['classes_names'],
                                       'put': options['id']
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_text_segmentation(paths_list: list, **options: dict) -> dict:

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

        text_list: dict = {}
        text_segm_data: dict = {}
        open_tags: list = options['parameters']['open_tags'].split(' ')
        close_tags: list = options['parameters']['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        length = options['parameters']['length'] if\
            options['parameters']['text_mode'] == LayerTextModeChoice.length_and_step else\
            options['parameters']['max_words']

        for path in paths_list:
            text_file = read_text(file_path=path, lower=True, del_symbols=options['parameters']['filters'], split=' ',
                                  open_symbol=open_symbol, close_symbol=close_symbol)
            if text_file:
                text_list[path] = get_samples(text_file, open_tags, close_tags)

        for key, value in sorted(text_list.items()):
            if options['parameters']['text_mode'] == LayerTextModeChoice.completely:
                text_segm_data[';'.join([key, f'[0-{options["parameters"]["max_words"]}]'])] =\
                    value[:options['parameters']['max_words']]
            elif options['parameters']['text_mode'] == LayerTextModeChoice.length_and_step:
                max_length = len(value)
                cur_step = 0
                stop_flag = False
                while not stop_flag:
                    text_segm_data[';'.join([key, f'[{cur_step}-{cur_step + length}]'])] = value[
                                                                                           cur_step:cur_step + length]
                    cur_step += options['parameters']['step']
                    if cur_step + length > max_length:
                        stop_flag = True

        instructions = {'instructions': text_segm_data,
                        'parameters': {'num_classes': len(open_tags),
                                       'classes_names': open_tags,
                                       'put': options['id'],
                                       'length': length
                                       }
                        }

        return instructions


    @staticmethod
    def cut_image(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        for elem in paths_list:
            os.makedirs(os.path.join(tmp_folder, f'{options["put"]}_image', os.path.basename(os.path.dirname(elem))),
                        exist_ok=True)
            shutil.copyfile(elem,
                            os.path.join(tmp_folder, f'{options["put"]}_image', os.path.basename(os.path.dirname(elem)),
                                         os.path.basename(elem)))

        if dataset_folder:
            if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_image')):
                shutil.move(os.path.join(tmp_folder, f'{options["put"]}_image'), dataset_folder)

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_video(paths_list: list, tmp_folder=None, dataset_folder=None, **options):

        def add_frames(video_array, fill_mode, frames_to_add, total_frames):

            frames: np.ndarray = np.array([])

            if fill_mode == LayerVideoFillModeChoice.black_frames:
                frames = np.zeros((frames_to_add, *orig_shape, 3), dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.average_value:
                mean = np.mean(video_array, axis=0, dtype='uint16')
                frames = np.full((frames_to_add, *mean.shape), mean, dtype='uint8')
            elif fill_mode == LayerVideoFillModeChoice.last_frames:
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
            os.makedirs(os.path.join(tmp_folder, f'{options["put"]}_video', os.path.basename(os.path.dirname(elem))),
                        exist_ok=True)
            path, slicing = elem.split(';')
            slicing = [int(x) for x in slicing[1:-1].split('-')]
            name, ext = os.path.splitext(os.path.basename(path))
            cap = cv2.VideoCapture(path)
            cap.set(1, slicing[0])
            orig_shape = (int(cap.get(3)), int(cap.get(4)))
            frames_count = int(cap.get(7))
            frames_number = 0
            save_path = os.path.join(tmp_folder, f'{options["put"]}_video', os.path.basename(os.path.dirname(elem)),
                                     f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
            instructions_paths.append(save_path)
            output_movie = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), orig_shape)
            stop_flag = False
            while not stop_flag:
                ret, frame = cap.read()
                frames_number += 1
                output_movie.write(frame)
                if options['video_mode'] == 'completely' and options['max_frames'] > frames_count and ret or\
                        options['video_mode'] == 'length_and_step' and options['length'] > frames_count and ret:
                    tmp_array.append(frame)
                if not ret or frames_number > frames_count:
                    stop_flag = True
            if options['video_mode'] == 'completely' and options['max_frames'] > frames_count or\
                    options['video_mode'] == 'length_and_step' and options['length'] > frames_count:
                frames_to_add = add_frames(video_array=np.array(tmp_array),
                                           fill_mode=options['fill_mode'],
                                           frames_to_add=options['max_frames'] - frames_count,
                                           total_frames=options['max_frames'])
                for arr in frames_to_add:
                    output_movie.write(arr)

            output_movie.release()

        if dataset_folder:
            if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_video')):
                shutil.move(os.path.join(tmp_folder, f'{options["put"]}_video'), dataset_folder)

        instructions = {'instructions': instructions_paths,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_audio(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        instructions_paths = []
        for elem in paths_list:
            path, slicing = elem.split(';')
            name, ext = os.path.splitext(os.path.basename(path))
            slicing = [float(x) for x in slicing[1:-1].split('-')]
            os.makedirs(os.path.join(tmp_folder, f'{options["put"]}_audio', os.path.basename(os.path.dirname(path))),
                        exist_ok=True)
            audio = AudioSegment.from_file(path, start_second=slicing[0], duration=slicing[1])
            save_path = os.path.join(tmp_folder, f'{options["put"]}_audio', os.path.basename(os.path.dirname(path)),
                                     f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
            audio.export(save_path)
            instructions_paths.append(save_path)

        if dataset_folder:
            if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_audio')):
                shutil.move(os.path.join(tmp_folder, f'{options["put"]}_audio'), dataset_folder)

        instructions = {'instructions': instructions_paths,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_text(paths_list: dict, tmp_folder=None, dataset_folder=None, **options: dict):

        text_list = []
        for elem in sorted(paths_list.keys()):
            # path, _ = elem.split(';')
            # name, ext = os.path.splitext(os.path.basename(path))
            # os.makedirs(os.path.join(tmp_folder, f'{options["put"]}_text', os.path.basename(os.path.dirname(path))), exist_ok=True)
            # shutil.copyfile(path, os.path.join(tmp_folder, f'{options["put"]}_text', os.path.basename(os.path.dirname(path)), os.path.basename(path)))
            text_list.append(paths_list[elem])

        # if dataset_folder:
        # if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_text')):
        # shutil.move(os.path.join(tmp_folder, f'{options["put"]}_text'), dataset_folder)

        text_list = []
        for key in sorted(paths_list.keys()):
            text_list.append(paths_list[key])
        instructions = {'instructions': text_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_classification(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_regression(number_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        instructions = {'instructions': number_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_segmentation(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        for elem in paths_list:
            os.makedirs(
                os.path.join(tmp_folder, f'{options["put"]}_segmentation', os.path.basename(os.path.dirname(elem))),
                exist_ok=True)
            shutil.copyfile(elem, os.path.join(tmp_folder, f'{options["put"]}_segmentation',
                                               os.path.basename(os.path.dirname(elem)), os.path.basename(elem)))

        if dataset_folder:
            if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_segmentation')):
                shutil.move(os.path.join(tmp_folder, f'{options["put"]}_segmentation'), dataset_folder)

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_text_segmentation(paths_list: dict, tmp_folder=None, dataset_folder=None, **options: dict):

        text_list = []
        for elem in sorted(paths_list.keys()):
            # path, slicing = elem.split(';')
            # name, ext = os.path.splitext(os.path.basename(path))
            # os.makedirs(os.path.join(tmp_folder, f'{options["put"]}_text_segmentation', os.path.basename(os.path.dirname(path))), exist_ok=True)
            # shutil.copyfile(path, os.path.join(tmp_folder, f'{options["put"]}_text_segmentation', os.path.basename(os.path.dirname(path)), os.path.basename(path)))
            text_list.append(paths_list[elem])

        # if dataset_folder:
        # if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_text_segmentation')):
        # shutil.move(os.path.join(tmp_folder, f'{options["put"]}_text_segmentation'), dataset_folder)

        text_list = []
        for key in sorted(paths_list.keys()):
            text_list.append(paths_list[key])

        instructions = {'instructions': text_list,
                        'parameters': options}

        return instructions


    @staticmethod
    def create_image(image_path: str, **options) -> dict:

        img = load_img(image_path)
        array = img_to_array(img, dtype=np.uint8)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_video(video_path: str, **options) -> dict:

        array = []
        slicing = [int(x) for x in video_path[video_path.index('[') + 1:video_path.index(']')].split('-')]
        frames_count = slicing[1] - slicing[0]
        cap = cv2.VideoCapture(video_path)
        # cap.set(1, slicing[0])
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

        if options['scaler'] != LayerScalerVideoChoice.no_scaler and options['object_scaler']:
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_audio(audio_path: str, **options) -> dict:

        array = []
        parameter = options['parameter']
        sample_rate = options['sample_rate']
        y, sr = librosa_load(path=audio_path, sr=options.get('sample_rate'), res_type='kaiser_best')
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

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_text(text: str, **options) -> dict:

        instructions = {'instructions': text,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_classification(class_name: str, **options) -> dict:

        index = options['classes_names'].index(os.path.basename(class_name))
        if options['one_hot_encoding']:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')
        index = np.array(index)

        instructions = {'instructions': index,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_regression(index: int, **options) -> dict:

        instructions = {'instructions': np.array(index),
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
        array = img_to_array(img, dtype=np.uint8)
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
    def preprocess_image(array: np.ndarray, **options) -> np.ndarray:

        array = cv2.resize(array, (options['width'], options['height']))

        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('object_scaler'):
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

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

        if options['scaler'] != LayerScalerVideoChoice.no_scaler and options.get('object_scaler'):
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def preprocess_audio(array: np.ndarray, **options) -> np.ndarray:

        if options['scaler'] != LayerScalerAudioChoice.no_scaler and options.get('object_scaler'):
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def preprocess_text(text: str, **options) -> np.ndarray:

        array = []
        text = text.split(' ')
        words_to_add = []

        if options['prepare_method'] == LayerPrepareMethodChoice.embedding:
            array = options['object_tokenizer'].texts_to_sequences([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.bag_of_words:
            array = options['object_tokenizer'].texts_to_matrix([text])[0]
        elif options['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
            for word in text:
                try:
                    array.append(options['object_word2vec'][word])
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
    def preprocess_classification(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def preprocess_regression(array: np.ndarray, **options) -> np.ndarray:

        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('object_scaler'):
            orig_shape = array.shape
            array = options['object_scaler'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def preprocess_segmentation(array: np.ndarray, **options) -> np.ndarray:

        return array

    @staticmethod
    def preprocess_text_segmentation(array: np.ndarray, **options) -> np.ndarray:

        return array
