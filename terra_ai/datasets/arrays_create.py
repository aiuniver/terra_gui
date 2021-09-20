from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerScalerVideoChoice, LayerPrepareMethodChoice
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
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import utils
from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum


class CreateArray(object):

    @staticmethod
    def instructions_image(paths_list: list, **options: dict) -> dict:

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
    def instructions_dataframe(_, **options: dict) -> dict:

        instructions = {"instructions": {}, 'parameters': options["parameters"]}
        instructions['parameters']['put'] = options["id"]
        if options["parameters"]['length']:
            if options["parameters"]["transpose"]:
                general_df = pd.read_csv(
                    os.path.join(options["parameters"]["sources_paths"][0]),
                    sep=options["parameters"]["separator"]).T
                general_df.columns = general_df.iloc[0]
                general_df.drop(general_df.index[[0]], inplace=True)
                general_df.index = range(0, len(general_df))
                for i in options["parameters"]["cols_names"][0]:
                    general_df = general_df.astype(
                        {general_df.columns[i]: np.float}, errors="ignore")
                df = general_df.iloc[:, options["parameters"]["cols_names"][0]]
            else:
                df = pd.read_csv(options["parameters"]["sources_paths"][0],
                                 usecols=options["parameters"]['cols_names'],
                                 sep=options["parameters"]["separator"])
            instructions['parameters']["timeseries"] = True
        else:
            y_col = options["parameters"]['y_cols']
            if options["parameters"]["pad_sequences"] or options["parameters"]["xlen_step"]:
                if options["parameters"]["pad_sequences"]:
                    example_length = options["parameters"]["example_length"]
                    tmp_df = pd.read_csv(options["parameters"]["sources_paths"][0],
                                         usecols=range(0, example_length + 1),
                                         sep=options["parameters"]["separator"])
                    tmp_df.sort_values(by=tmp_df.columns[0], ignore_index=True, inplace=True)
                    tmp_df.fillna(0, inplace=True)
                    df = tmp_df.iloc[:, range(1, example_length + 1)]

                elif options["parameters"]["xlen_step"]:
                    xlen = options["parameters"]["xlen"]
                    step_len = options["parameters"]["step_len"]

                    df = pd.read_csv(options["parameters"]["sources_paths"][0],
                                     sep=options["parameters"]["separator"])
                    df.sort_values(by=df.columns[0], ignore_index=True, inplace=True)
                    df = df.iloc[:, 1:]
                    xlen_array = []
                    for i in range(len(df)):
                        subdf = df.iloc[i, :]
                        subdf = subdf.dropna().values.tolist()
                        for j in range(0, len(subdf), step_len):
                            if len(subdf[j: j + xlen]) < xlen:
                                xlen_array.append(subdf[-xlen:])
                            else:
                                xlen_array.append(subdf[j: j + xlen])
                    tmp_dict = {}
                    for i in range(xlen):
                        tmp_dict.update({i: np.array(xlen_array)[:, i]})
                    df = pd.DataFrame(tmp_dict)
                instructions["parameters"]["scaler"] = options["parameters"]["scaler"]
            else:
                tmp_df = pd.read_csv(options["parameters"]["sources_paths"][0],
                                     sep=options["parameters"]["separator"], nrows=1)
                df = pd.read_csv(options["parameters"]["sources_paths"][0],
                                 usecols=options["parameters"]["cols_names"] + y_col,
                                 sep=options["parameters"]["separator"])
                sort_col = tmp_df.columns.tolist()[y_col[0]]
                x_cols = []
                for idx in options["parameters"]["cols_names"]:
                    x_cols.append(tmp_df.columns.tolist()[idx])
                df.sort_values(by=sort_col, ignore_index=True, inplace=True)
                df = df.loc[:, x_cols]
        instructions["instructions"] = df.to_dict()

        if options["parameters"]["Categorical_cols"]:
            tmp_lst = options["parameters"]["Categorical_cols"]
            instructions["parameters"]["Categorical_cols"] = {}
            instructions["parameters"]["Categorical_cols"]["lst_cols"] = tmp_lst
            for i in instructions["parameters"]["Categorical_cols"]["lst_cols"]:
                instructions["parameters"]["Categorical_cols"][f"col_{i}"] = list(set(df.iloc[:, i]))

        if options["parameters"]["Categorical_ranges_cols"]:
            tmp_lst = options["parameters"]["Categorical_ranges_cols"]
            instructions["parameters"]["Categorical_ranges_cols"] = {}
            instructions["parameters"]["Categorical_ranges_cols"]["lst_cols"] = tmp_lst
            for i in range(len(tmp_lst)):
                if len(list(options["parameters"]["cat_cols"].values())[i].split(" ")) == 1:
                    border = max(df.iloc[:, tmp_lst[i]]) / int(list(options["parameters"]["cat_cols"].values())[i])
                    instructions["parameters"]["Categorical_ranges_cols"][f"col_{tmp_lst[i]}"] = np.linspace(
                        border, max(df.iloc[:, tmp_lst[i]]),
                        int(list(options["parameters"]["cat_cols"].values())[i])).tolist()
                else:
                    instructions["parameters"]["Categorical_ranges_cols"][f"col_{tmp_lst[i]}"] = \
                        list(options["parameters"]["cat_cols"].values())[i].split(" ")

        if options["parameters"]["one_hot_encoding_cols"]:
            tmp_lst = options["parameters"]["one_hot_encoding_cols"]
            instructions["parameters"]["one_hot_encoding_cols"] = {}
            instructions["parameters"]["one_hot_encoding_cols"]["lst_cols"] = tmp_lst
            for i in instructions["parameters"]["one_hot_encoding_cols"]["lst_cols"]:
                if options["parameters"]["Categorical_ranges_cols"] and (
                        i in instructions["parameters"]["Categorical_ranges_cols"]["lst_cols"]):
                    instructions["parameters"]["one_hot_encoding_cols"][f"col_{i}"] = len(
                        instructions["parameters"]["Categorical_ranges_cols"][f"col_{i}"])
                else:
                    instructions["parameters"]["one_hot_encoding_cols"][f"col_{i}"] = len(
                        set(df.iloc[:, i]))

        return instructions

    @staticmethod
    def instructions_classification(paths_list: list, **options: dict) -> dict:
        type_processing = options['parameters']['type_processing']
        if options["parameters"]["xlen_step"]:
            xlen = options["parameters"]["xlen"]
            step_len = options["parameters"]["step_len"]

            df = pd.read_csv(options["parameters"]["sources_paths"][0],
                             sep=options["parameters"]["separator"])
            df.sort_values(by=df.columns[0], ignore_index=True, inplace=True)
            classes_names = df[df.columns[0]].tolist()
            df = df.iloc[:, 1:]
            y_class = []
            for i in range(len(df)):
                subdf = df.iloc[i, :]
                subdf = subdf.dropna().values.tolist()
                for j in range(0, len(subdf), step_len):
                    if len(subdf[j: j + xlen]) < xlen:
                        y_class.append(classes_names[i])
                    else:
                        y_class.append(classes_names[i])
            paths_list = y_class

        elif os.path.isfile(options['parameters']['sources_paths'][0]) and \
                options['parameters']['sources_paths'][0].endswith('.csv'):
            file_name = options['parameters']['sources_paths'][0]
            data = pd.read_csv(file_name, usecols=options['parameters']['cols_names'],
                                     sep=options["parameters"]["separator"])
            data.sort_values(by=data.columns[0], ignore_index=True, inplace=True)
            column = data.iloc[:, 0].to_list()

            if type_processing == "categorical":
                classes_names = []
                for elem in column:
                    if elem not in classes_names:
                        classes_names.append(elem)
            else:
                if len(options['parameters']["ranges"].split(" ")) == 1:
                    border = max(column) / int(options['parameters']["ranges"])
                    classes_names = np.linspace(border, max(column),
                                                int(options['parameters']["ranges"])).tolist()
                else:
                    classes_names = options['parameters']["ranges"].split(" ")
        else:
            classes_names = sorted([os.path.basename(elem) for elem in options['parameters']['sources_paths']])

        instructions = {'instructions': paths_list,
                        'parameters': {"one_hot_encoding": options['parameters']['one_hot_encoding'],
                                       "classes_names": classes_names,
                                       "num_classes": len(classes_names),
                                       'put': options['id'],
                                       "type_processing": type_processing}
                        }

        return instructions

    @staticmethod
    def instructions_regression(number_list: list, **options: dict) -> dict:

        instructions = {'instructions': number_list,
                        'parameters': options["parameters"]}
        instructions['parameters']['put'] = options["id"]

        return instructions

    @staticmethod
    def instructions_segmentation(paths_list: list, **options: dict) -> dict:

        instructions = {'instructions': paths_list,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': len(options['classes_names']),
                                       'height': options['height'],
                                       'width': options['width'],
                                       'classes_colors': [Color(color).as_rgb_tuple() for color in
                                                          options['classes_colors']],
                                       'classes_names': options['classes_names'],
                                       'cols_names': options['cols_names'],
                                       'put': options['put']
                                       }
                        }

        return instructions

    @staticmethod
    def instructions_text_segmentation(paths_list: list, **options: dict) -> dict:

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
                    # print(word)

            return indexes

        text_list: dict = {}
        text_segm_data: dict = {}
        open_tags: list = options['parameters']['open_tags'].split(' ')
        close_tags: list = options['parameters']['close_tags'].split(' ')
        open_symbol = open_tags[0][0]
        close_symbol = close_tags[0][-1]
        length = options['parameters']['length'] if \
            options['parameters']['text_mode'] == LayerTextModeChoice.length_and_step else \
            options['parameters']['max_words']

        for path in paths_list:
            text_file = read_text(file_path=path, lower=True, del_symbols=options['parameters']['filters'], split=' ',
                                  open_symbol=open_symbol, close_symbol=close_symbol)
            if text_file:
                text_list[path] = get_samples(text_file, open_tags, close_tags)

        for key, value in sorted(text_list.items()):
            if options['parameters']['text_mode'] == LayerTextModeChoice.completely:
                text_segm_data[';'.join([key, f'[0-{options["parameters"]["max_words"]}]'])] = \
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
    def instructions_timeseries(_, **options: dict) -> dict:

        instructions = {"instructions": {}, "parameters": options["parameters"]}
        instructions["parameters"]["put"] = options["id"]
        if options["parameters"]["transpose"]:
            tmp_df_ts = pd.read_csv(options["parameters"]["sources_paths"][0], sep=options["parameters"]["separator"]).T
            tmp_df_ts.columns = tmp_df_ts.iloc[0]
            tmp_df_ts.drop(tmp_df_ts.index[[0]], inplace=True)
            tmp_df_ts.index = range(0, len(tmp_df_ts))
            for i in instructions["parameters"]["cols_names"]:
                tmp_df_ts = tmp_df_ts.astype({i: np.float}, errors="ignore")
            y_subdf = tmp_df_ts.loc[:, instructions["parameters"]["cols_names"]]
        else:
            y_subdf = pd.read_csv(
                options["parameters"]["sources_paths"][0], sep=options["parameters"]["separator"],
                usecols=instructions["parameters"]["cols_names"])

        if options["parameters"]['trend']:
            instructions['parameters']['classes_names'] = ["Не изменился", "Вверх", "Вниз"]
            instructions['parameters']['num_classes'] = 3
        instructions["instructions"] = y_subdf.to_dict()
        return instructions

    @staticmethod
    def instructions_object_detection(paths_list: list, **options: dict) -> dict:

        instructions = {'instructions': paths_list,
                        'parameters': {'yolo': options['parameters']['yolo'],
                                       'num_classes': options['parameters']['num_classes'],
                                       'classes_names': options['parameters']['classes_names'],
                                       'put': options['id']}}

        return instructions

    @staticmethod
    def cut_image(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict):

        for elem in paths_list:
            os.makedirs(os.path.join(tmp_folder, f'{options["cols_names"]}', os.path.basename(os.path.dirname(elem))), exist_ok=True)
            shutil.copyfile(elem, os.path.join(tmp_folder, f'{options["cols_names"]}', os.path.basename(os.path.dirname(elem)), os.path.basename(elem)))

        if dataset_folder:
            if os.path.isdir(os.path.join(dataset_folder, f'{options["cols_names"]}')):
                shutil.rmtree(os.path.join(dataset_folder, f'{options["cols_names"]}'))
            shutil.move(os.path.join(tmp_folder, f'{options["cols_names"]}'), dataset_folder)

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
                if options['video_mode'] == 'completely' and options['max_frames'] > frames_count and ret or \
                        options['video_mode'] == 'length_and_step' and options['length'] > frames_count and ret:
                    tmp_array.append(frame)
                if not ret or frames_number > frames_count:
                    stop_flag = True
            if options['video_mode'] == 'completely' and options['max_frames'] > frames_count or \
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
            text_list.append(paths_list[elem])

        text_list = []
        for key in sorted(paths_list.keys()):
            text_list.append(paths_list[key])
        instructions = {'instructions': text_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_dataframe(paths_list: dict, tmp_folder=None, dataset_folder=None, **options: dict):

        instructions = {'instructions': paths_list,
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
            os.makedirs(os.path.join(tmp_folder, f'{options["cols_names"]}', os.path.basename(os.path.dirname(elem))), exist_ok=True)
            shutil.copyfile(elem, os.path.join(tmp_folder, f'{options["cols_names"]}', os.path.basename(os.path.dirname(elem)), os.path.basename(elem)))

        if dataset_folder:
            if os.path.isdir(os.path.join(dataset_folder, f'{options["cols_names"]}')):
                shutil.rmtree(os.path.join(dataset_folder, f'{options["cols_names"]}'))
            shutil.move(os.path.join(tmp_folder, f'{options["cols_names"]}'), dataset_folder)

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_text_segmentation(paths_list: dict, tmp_folder=None, dataset_folder=None, **options: dict):

        text_list = []
        for elem in sorted(paths_list.keys()):
            text_list.append(paths_list[elem])

        text_list = []
        for key in sorted(paths_list.keys()):
            text_list.append(paths_list[key])

        instructions = {'instructions': text_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_timeseries(paths_list: dict, tmp_folder=None, dataset_folder=None, **options: dict):

        instructions = {'instructions': paths_list,
                        'parameters': options}

        return instructions

    @staticmethod
    def cut_object_detection(paths_list: list, tmp_folder=None, dataset_folder=None, **options: dict) -> dict:

        for elem in paths_list:
            os.makedirs(
                os.path.join(tmp_folder, f'{options["put"]}_object_detection', os.path.basename(os.path.dirname(elem))),
                exist_ok=True)
            shutil.copyfile(elem,
                            os.path.join(tmp_folder, f'{options["put"]}_object_detection',
                                         os.path.basename(os.path.dirname(elem)),
                                         os.path.basename(elem)))

        if dataset_folder:
            if not os.path.isdir(os.path.join(dataset_folder, f'{options["put"]}_object_detection')):
                shutil.move(os.path.join(tmp_folder, f'{options["put"]}_object_detection'), dataset_folder)

        instructions = {'instructions': paths_list,
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
    def create_dataframe(row, **options) -> dict:

        instructions = {'instructions': row,
                        'parameters': options}

        return instructions

    @staticmethod
    def create_classification(class_name: str, **options) -> dict:

        if options['type_processing'] == 'categorical':
            if '.trds' in str(class_name):
                index = options['classes_names'].index(os.path.basename(class_name))
            else:
                index = options['classes_names'].index(class_name)
        else:
            for i, cl_name in enumerate(options['classes_names']):
                if class_name <= int(cl_name):
                    index = i
                    break
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
    def create_timeseries(row, **options) -> dict:

        instructions = {'instructions': row,
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
        yolo_anchors = None

        if options['yolo'] == 'v3':
            yolo_anchors = [[[10, 13], [16, 30], [33, 23]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[116, 90], [156, 198], [373, 326]]]
        elif options['yolo'] == 'v4':
            yolo_anchors = [[[12, 16], [19, 36], [40, 28]],
                            [[36, 75], [76, 55], [72, 146]],
                            [[142, 110], [192, 243], [459, 401]]]

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

        instructions = {'instructions': [np.array(label_sbbox, dtype='float32'), np.array(sbboxes, dtype='float32'),
                                         np.array(label_mbbox, dtype='float32'), np.array(mbboxes, dtype='float32'),
                                         np.array(label_lbbox, dtype='float32'), np.array(lbboxes, dtype='float32')],
                        'parameters': options}

        return instructions

    @staticmethod
    def preprocess_image(array: np.ndarray, **options) -> np.ndarray:

        array = cv2.resize(array, (options['width'], options['height']))
        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
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
    def preprocess_dataframe(row: np.ndarray, **options) -> np.ndarray:
        length = options['length'] if 'timeseries' in options.keys() else 1
        if length == 1:
            row = row if options['xlen_step'] else [row]
        if options['scaler'] != 'no_scaler':
            row = np.array(row)
            orig_shape = row.shape
            array = options['object_scaler'].transform(row.reshape(-1, 1))
            array = array.reshape(orig_shape)
            return array

        if options['MinMaxScaler_cols']:
            for j in range(length):
                for i in options['MinMaxScaler_cols']:
                    row[j][i] = options['object_scaler'][f'col_{i}'].transform(
                        np.array(row[j][i]).reshape(-1, 1)).tolist()[0][0]

        if options['StandardScaler_cols']:
            for j in range(length):
                for i in options['StandardScaler_cols']:
                    row[j][i] = options['object_scaler'][f'col_{i}'].transform(
                        np.array(row[j][i]).reshape(-1, 1)).tolist()[0][0]

        if options['Categorical_cols']:
            for j in range(length):
                for i in options['Categorical_cols']['lst_cols']:
                    row[j][i] = list(options['Categorical_cols'][f'col_{i}']).index(row[j][i])

        if options['Categorical_ranges_cols']:
            for j in range(length):
                for i in options['Categorical_ranges_cols']['lst_cols']:
                    for k in range(len(options['Categorical_ranges_cols'][f'col_{i}'])):
                        if row[j][i] <= int(options['Categorical_ranges_cols'][f'col_{i}'][k]):
                            row[j][i] = k
                            break

        if options['one_hot_encoding_cols']:
            for j in range(length):
                for i in options['one_hot_encoding_cols']['lst_cols']:
                    row[j][i] = utils.to_categorical(row[j][i], options['one_hot_encoding_cols'][f'col_{i}'],
                                                     dtype='uint8').tolist()

        if type(row) != list:
            row = row.tolist()

        if options['xlen_step']:
            array = np.array(row)
        else:
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

    @staticmethod
    def preprocess_timeseries(row: np.ndarray, **options) -> np.ndarray:
        if options["trend"]:
            first_value = row[0][0]
            second_value = row[1][0]

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
            # if options['one_hot_encoding']:
            #     tmp_unique = list(set(trends))
            #     for i in range(len(trends)):
            #         trends[i] = tmp_unique.index(trends[i])
        else:
            array = row
            if options['scaler'] != 'no_scaler':
                orig_shape = row.shape
                array = options['object_scaler'].transform(row.reshape(-1, 1))
                array = array.reshape(orig_shape)
        return array

    @staticmethod
    def preprocess_object_detection(array: list, **options):

        return array
