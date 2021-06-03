from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from time import time
from PIL import Image, ImageColor
from inspect import getmembers, signature
from librosa import load as librosaload
import librosa.feature as librosafeature
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import re
import pymorphy2
import shutil
from gensim.models import word2vec
from tqdm.notebook import tqdm
from io import open as ioopen
from terra_ai.guiexchange import Exchange
import ipywidgets as widgets
import dill
import configparser
import joblib
from ast import literal_eval
from urllib import request
from tempfile import mkdtemp
from IPython.display import display
from datetime import datetime
from pytz import timezone
import json

# import cv2

__version__ = 0.317

tr2dj_obj = Exchange()


class DTS(object):

    def __init__(self, path=mkdtemp(), exch_obj=tr2dj_obj):

        self.Exch = exch_obj
        self.django_flag = False
        if self.Exch.property_of != 'TERRA':
            self.django_flag = True

        self.divide_ratio = [(0.8, 0.2), (0.8, 0.1, 0.1)]
        self.file_folder: str = ''
        self.save_path: str = path
        self.name: str = ''
        self.source: str = ''
        self.tags: dict = {}
        self.source_datatype: str = ''
        self.source_shape: dict = {}
        self.input_datatype: str = ''
        self.input_shape: dict = {}
        self.output_datatype: dict = {}
        self.output_shape: dict = {}
        self.one_hot_encoding = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.language: str = ''
        self.dts_prepared: bool = False
        self.task_type: dict = {}
        self.user_parameters: dict = {}
        self.user_tags: list = []

        self.X: dict = {}
        self.Y: dict = {}
        self.x_Scaler: dict = {}
        self.y_Scaler: dict = {}
        self.tokenizer: dict = {}
        self.word2vec: dict = {}
        self.df: dict = {}
        self.tsgenerator: dict = {}

        self.y_Cls: np.array
        self.peg: list = []

        pass

    def call_method(self, name: list) -> dict:

        func_params = {}
        for tupl in getmembers(self):
            if tupl[0] == name:
                sig = signature(tupl[1])
                for param in sig.parameters.values():
                    func_params[param.name] = param.default
                break

        return func_params

    def get_datasets_dict(self) -> dict:
        # ['болезни', 'жанры_музыки', 'трафик', 'диалоги']

        datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters', 'sber',
                    'автомобили', 'автомобили_3', 'самолеты', 'губы', 'заболевания', 'договоры', 'умный_дом', 'трейдинг',
                    'квартиры']

        datasets_dict = {}
        for data in datasets:
            datasets_dict[data] = [self._set_tag(data), self._set_language(data), self._set_source(data)]

        return datasets_dict

    def get_parameters_dict(self) -> dict:

        parameters = {}
        list_of_params = ['images', 'text', 'audio', 'dataframe'] + ['classification', 'segmentation', 'text_segmentation',
                                                                     'regression', 'timeseries']

        for elem in list_of_params:
            temp = {}
            for key, value in self.call_method(elem).items():
                if type(value) == list:
                    if key == 'folder_name' and self.file_folder or key == 'file_name' and self.file_folder:
                        value += os.listdir(self.file_folder)
                    temp[key] = {'type': type(value[0]).__name__,
                                 'default': value[0],
                                 'list': True,
                                 'available': value}
                else:
                    temp[key] = {'type': type(value).__name__,
                                 'default': value}
            parameters[elem] = temp

        return parameters

    def load_dataset(self) -> None:

        def on_button_clicked(b):

            if load_tab.selected_index == 0:
                self.load_data(name=zip_name.value, link=None, mode='google_drive')
            elif load_tab.selected_index == 1:
                self.load_data(name=None, link=url_google.value, mode='url')
            elif load_tab.selected_index == 2:
                self.prepare_dataset(dataset_name=terra_dataset.value)

            pass

        button = widgets.Button(description='Загрузить', disabled=False, button_style='', tooltip='Загрузить датасет',
                                icon='check')

        #Первая вкладка
        if os.getcwd() == '/content':
            filelist = os.listdir('/content/drive/MyDrive/TerraAI/datasets/sources')
            if not filelist:
                filelist = ['Нет файлов']
        else: # Для тестирования на локалке
            filelist = ['Гугл диск недоступен.']
        zip_name = widgets.Dropdown(options=filelist, value=filelist[0], description='Файл', disabled=False)
        google_drive = widgets.VBox([zip_name, button])

        #Вторая вкладка
        url_google = widgets.Text(value='', placeholder='https://', description='URL:', disabled=False)
        vbox_download = widgets.VBox([url_google, button])

        # Третья вкладка
        datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters', 'sber',
         'автомобили', 'автомобили_3', 'самолеты', 'губы', 'заболевания', 'договоры', 'умный_дом', 'трейдинг',
         'квартиры']
        terra_dataset = widgets.RadioButtons(options=datasets, value=datasets[0], description='Список баз:', disabled=False)
        vbox_terra_dataset = widgets.VBox([terra_dataset, button])

        load_tab = widgets.Tab()
        load_tab.children = [google_drive, vbox_download, vbox_terra_dataset]

        load_tab.set_title(title='Google Drive', index=0)
        load_tab.set_title(title='URL-ссылка', index=1)
        load_tab.set_title(title='Terra-ai', index=2)


        button.on_click(on_button_clicked)

        display(load_tab)

        pass

    def create_dataset(self, inputs: int, outputs: int) -> None:

        def build_widget(x):

            input = self.call_method(x)

            if 'folder_name' in input.keys() and self.file_folder != '':
                input['folder_name'] = [''] + sorted(os.listdir(f'{self.file_folder}'))
            elif 'file_name' in input.keys() and self.file_folder != '':
                input['file_name'] = [''] + sorted(os.listdir(f'{self.file_folder}'))

            def get_text(key, value):
                text_widget = widgets.Text(value=value, placeholder='', description=key, disabled=False)
                return text_widget

            def get_int(key, value):
                integer_widget = widgets.IntText(value=value, description=key, disabled=False)
                return integer_widget

            def get_float(key, value):
                float_widget = widgets.FloatText(value=value, description=key, disabled=False)
                return float_widget

            def get_dropdown(key, value):
                dropdown_widget = widgets.Dropdown(options=value, value=value[0], description=key, disabled=False)
                return dropdown_widget

            def get_checkbox(key, value):
                checkbox_widget = widgets.Checkbox(value=value, description=key, disabled=False)
                return checkbox_widget

            def manual_input(x):

                blocks = []
                for i in range(x):
                    color_name = widgets.Text(value='', placeholder='', description=f'Класс {i + 1}', disabled=False)
                    color_choose = widgets.ColorPicker(concise=False, description='Цвет', value='#FFFFFF',
                                                       disabled=False)
                    blocks.append(widgets.VBox([color_name, color_choose]))
                color_pick = widgets.VBox(blocks)
                display(color_pick)

                return color_pick

            def get_color(x):

                def find_classes(b):

                    def rgb_to_hex(rgb):
                        return '%02x%02x%02x' % rgb

                    for i in range(len(list_of_outputs)):
                        if list_of_outputs[i].children[3].kwargs['x'] == 'segmentation':
                            folder_name = list_of_outputs[i].children[3].result.children[0].value
                            mask_range = list_of_outputs[i].children[3].result.children[1].value

                    color_list = self._find_colors(folder_name, auto_classes.value, mask_range)
                    color_widgets = []
                    for i, color in enumerate(color_list):
                        class_name = widgets.Text(value='', placeholder='', description=f'Класс {i + 1}',
                                                  disabled=False)
                        col = widgets.ColorPicker(concise=False, description='Цвет',
                                                  value=f'#{rgb_to_hex(tuple(color))}',
                                                  disabled=False)
                        block = widgets.VBox([class_name, col])
                        color_widgets.append(block)
                    global col_widget
                    col_widget = widgets.VBox(color_widgets)
                    display(col_widget)

                    pass

                if x == 'Ручной ввод':
                    data = widgets.interactive(manual_input,
                                               x=widgets.IntText(value=1, description='Кол-во классов', disabled=False))
                    display(data)
                elif x == 'Автоматический поиск':
                    global auto_classes
                    auto_classes = widgets.IntText(value=1, description='Кол-во классов', disabled=False)
                    auto_button = widgets.Button(description='Поиск', disabled=False, button_style='', icon='')
                    auto_button.on_click(find_classes)
                    data = widgets.VBox([auto_classes, auto_button])
                    display(data)
                elif x == 'Файл аннотаций':
                    data = widgets.Dropdown(options=os.listdir(self.file_folder),
                                            value=os.listdir(self.file_folder)[0], description='Выберите файл')
                    display(data)

                return data

            list_of_widgets = []

            for i, param in enumerate(input.keys(), 1):

                if input[param] is None or isinstance(input[param], str):
                    str_wid = get_text(param, input[param])
                    list_of_widgets.append(str_wid)

                elif isinstance(input[param], bool):
                    wid = get_checkbox(param, input[param])
                    list_of_widgets.append(wid)

                elif isinstance(input[param], int):
                    wid = get_int(param, input[param])
                    list_of_widgets.append(wid)

                elif isinstance(input[param], float):
                    wid = get_float(param, input[param])
                    list_of_widgets.append(wid)

                elif isinstance(input[param], list):
                    wid = get_dropdown(param, input[param])
                    list_of_widgets.append(wid)

                elif isinstance(input[param], tuple):
                    for tupl in input[param]:
                        wid = get_int(param, tupl)
                        list_of_widgets.append(wid)

                elif isinstance(input[param], dict) and data_type_out.value == 'segmentation':
                    choice = widgets.Dropdown(options=['Ручной ввод', 'Автоматический поиск', 'Файл аннотаций'],
                                              value='Ручной ввод', description='Ввод классов', disabled=False)
                    data = widgets.interactive(get_color, x=choice)
                    list_of_widgets.append(data)

            vbox = widgets.VBox([widg for widg in list_of_widgets])
            display(vbox)
            return vbox

        def create_dataset(b):

            tags = {}
            task = {}
            for i in range(inputs):
                tags[f'input_{i + 1}'] = list_of_inputs[i].children[2].kwargs['x']
            for i in range(outputs):
                tags[f'output_{i + 1}'] = list_of_outputs[i].children[3].kwargs['x']
                task[f'output_{i + 1}'] = list_of_outputs[i].children[2].value

            parameters = {}
            parameters['name'] = dataset_name.value
            parameters['user_tags'] = dataset_hashtags.value.split(' ')
            parameters['preserve_sequence'] = sequence.value
            parameters['train_part'] = slider.value / 100
            parameters['val_part'] = slider2.value / 200
            parameters['test_part'] = slider2.value / 200

            input_dicts = {}
            output_dicts = {}
            for i in range(len(list_of_inputs)):
                dic = {}
                params = {}
                dic['name'] = list_of_inputs[i].children[1].value
                dic['tag'] = list_of_inputs[i].children[2].kwargs['x']
                for j in range(len(list_of_inputs[i].children[2].result.children)):
                    params[list_of_inputs[i].children[2].result.children[j].description] = \
                        list_of_inputs[i].children[2].result.children[j].value
                dic['parameters'] = params
                input_dicts[f'input_{i + 1}'] = dic
            for i in range(len(list_of_outputs)):
                dic = {}
                params = {}
                dic['name'] = list_of_outputs[i].children[1].value
                dic['task_type'] = list_of_outputs[i].children[2].value
                dic['tag'] = list_of_outputs[i].children[3].kwargs['x']
                if dic['tag'] == 'segmentation':
                    params['folder_name'] = list_of_outputs[i].children[3].result.children[0].value
                    params['mask_range'] = list_of_outputs[i].children[3].result.children[1].value
                    classes_dict = {}
                    if list_of_outputs[i].children[3].result.children[2].kwargs['x'] == 'Ручной ввод':
                        for j in range(list_of_outputs[0].children[3].result.children[2].result.kwargs['x']):
                            classes_dict[
                                list_of_outputs[0].children[3].result.children[2].result.result.children[j].children[
                                    0].value] = list(ImageColor.getcolor(
                                list_of_outputs[0].children[3].result.children[2].result.result.children[j].children[
                                    1].value, 'RGB'))
                    elif list_of_outputs[i].children[3].result.children[2].kwargs['x'] == 'Файл аннотаций':
                        classes_dict = self._find_colors(list_of_outputs[i].children[3].result.children[2].result.value, txt_file=True)
                    params['classes_dict'] = classes_dict
                else:
                    for j in range(len(list_of_outputs[i].children[3].result.children)):
                        params[list_of_outputs[i].children[3].result.children[j].description] = \
                        list_of_outputs[i].children[3].result.children[j].value
                dic['parameters'] = params
                output_dicts[f'output_{i + 1}'] = dic

            dataset_dict = {}
            dataset_dict['inputs'] = input_dicts
            dataset_dict['outputs'] = output_dicts
            dataset_dict['parameters'] = parameters
            self.prepare_user_dataset(dataset_dict)

            pass

        list_of_inputs = []
        list_of_outputs = []
        for i in range(1, inputs + 1):
            title = widgets.HTML(value=f"<b>{f'input_{i}':-^67}</b>", placeholder='', description='')
            name = widgets.Text(value='', placeholder='Название входа', description=f'input_{i}', disabled=False)
            data_type = widgets.Dropdown(
                options=[('Картинки', 'images'), ('Видео', 'video'), ('Текст', 'text'), ('Аудио', 'audio'),
                         ('Датафрейм', 'dataframe')], value='images', description='Тип данных')
            data = widgets.interactive(build_widget, x=data_type)

            wid = widgets.VBox([title, name, data])
            list_of_inputs.append(wid)

        for i in range(1, outputs + 1):
            title = widgets.HTML(value=f"<b>{f'output_{i}':-^67}</b>", placeholder='', description='')
            name = widgets.Text(value='', placeholder='Название выхода', description=f'output_{i}', disabled=False)
            task_type = widgets.Dropdown(options=[('Классификация', 'classification'), ('Регргессия', 'regression'),
                                                  ('Сегментация изображений', 'segmentation'),
                                                  ('Сегментация текстов', 'text_segmentation'),
                                                  ('Обнаружение объектов', 'object_detection'),
                                                  ('Автокодировщик', 'autoencoder'),
                                                  ('Генеративно-состязательная сеть', 'gan'),
                                                  ('Временные ряды', 'timeseries')], value='classification',
                                         description='Тип задачи')
            data_type_out = widgets.Dropdown(options=[('Картинки', 'images'), ('Текст', 'text'), ('Аудио', 'audio'),
                                                  ('Классификация', 'classification'), ('Регргессия', 'regression'),
                                                  ('Обнаружение объектов', 'object_detection'),
                                                  ('Сегментация изображений', 'segmentation'),
                                                  ('Сегментация текстов', 'text_segmentation'),
                                                  ('Временные ряды', 'timeseries')], value='classification',
                                         description='Тип данных')
            data = widgets.interactive(build_widget, x=data_type_out)

            wid = widgets.VBox([title, name, task_type, data])
            list_of_outputs.append(wid)

        def value_changed(change):
            slider2.value = 100 - change.new

        def value_changed_2(change):
            slider.value = 100 - change.new

        slider = widgets.IntSlider(description='Train:', value=80, step=1, min=5, max=95)
        slider2 = widgets.IntSlider(description='Val+test:', value=20, step=1, min=5, max=95)
        slider.observe(value_changed, 'value')
        slider2.observe(value_changed_2, 'value')

        dataset_name = widgets.Text(value='', description='Датасет', placeholder='Название датасета', disabled=False)
        dataset_hashtags = widgets.Text(value='', description='Пользовательские теги', placeholder='Теги через пробел', disabled=False)
        button = widgets.Button(description='Сформировать', disabled=False, button_style='', icon='check')
        button.on_click(create_dataset)
        first_row = widgets.HBox([slider, slider2])
        sequence = widgets.Checkbox(value=False, description='Сохранить последовательность', disabled=False)
        third_row = widgets.HBox([dataset_name, dataset_hashtags, button])

        display(widgets.HBox(list_of_inputs + list_of_outputs))
        display(widgets.VBox([first_row, sequence, third_row]))

        pass

    def create_custom_dataset(self, inputs: int, outputs: int, globals: dict) -> None:

        def send_arrays(b):

            full_val = len(set([bool(globals[f'xval_{i}'].value) for i in range(inputs)] + [bool(globals[f'yval_{i}'].value) for i in range(outputs)]))
            full_test = len(set([bool(globals[f'xtest_{i}'].value) for i in range(inputs)] + [bool(globals[f'ytest_{i}'].value) for i in range(outputs)]))
            if full_val != 1:
                assert full_val == 1, 'Колонка валидационной выборки заполнена не полностью.'
            if full_test != 1:
                assert full_test == 1, 'Колонка тестовой выборки заполнена не полностью.'

            print('Начало формирования массивов.')
            self.name = dataset_name.value
            self.user_tags = dataset_hashtags.value.split(' ')
            tags = {}
            task = {}
            for i in range(inputs):
                tags[f'input_{i+1}'] = globals[f'x_tag_{i}'].value
            for i in range(outputs):
                tags[f'output_{i+1}'] = globals[f'y_tag_{i}'].value
                task[f'output_{i+1}'] = globals[f'y_task_{i}'].value
            self.tags = tags
            self.task_type = task

            list_of_X = list_of_rows[:inputs]
            dic_of_X = {}
            for i in range(len(list_of_X)):
                if checkbox_split.value:
                    dic_of_X[list_of_X[i].children[0].description[:-1]] = {
                        'data_name': list_of_X[i].children[0].value,
                        'data': (globals[list_of_X[i].children[2].value], None, None)
                    }
                else:
                    values = []
                    for j in range(2, 5):
                        if list_of_X[i].children[j].value:
                            values.append(globals[list_of_X[i].children[j].value])
                        else:
                            values.append(None)
                    dic_of_X[list_of_X[i].children[0].description[:-1]] = {
                        'data_name': list_of_X[i].children[0].value,
                        'data': tuple(values)
                    }

            list_of_Y = list_of_rows[inputs:][:outputs]
            dic_of_Y = {}
            for i in range(len(list_of_Y)):
                if checkbox_split.value:
                    dic_of_Y[list_of_Y[i].children[0].description[:-1]] = {
                        'data_name': list_of_Y[i].children[0].value,
                        'data': (globals[list_of_Y[i].children[2].value], None, None)
                    }
                else:
                    values = []
                    for j in range(2, 5):
                        if list_of_Y[i].children[j].value:
                            values.append(globals[list_of_Y[i].children[j].value])
                        else:
                            values.append(None)
                    dic_of_Y[list_of_Y[i].children[0].description[:-1]] = {
                        'data_name': list_of_Y[i].children[0].value,
                        'data': tuple(values)
                    }

            ohe = {}
            for i in range(outputs):
                ohe[globals[f'y_name_{i}'].description[:-1]] = globals[f'ohe_{i}'].value

            x_scaler = {}
            for i in range(inputs):
                x_scaler[globals[f'x_name_{i}'].description[:-1]] = globals[f'scaler_x_{i}'].value

            y_scaler = {}
            for i in range(outputs):
                y_scaler[globals[f'y_name_{i}'].description[:-1]] = globals[f'scaler_y_{i}'].value
                if ohe[globals[f'y_name_{i}'].description[:-1]]:
                    y_scaler[globals[f'y_name_{i}'].description[:-1]] = 'Не применять'

            x_shape = {}
            for i in range(inputs):
                x_shape[globals[f'x_name_{i}'].description[:-1]] = globals[f'net_type_x_{i}'].value

            y_shape = {}
            for i in range(outputs):
                y_shape[globals[f'y_name_{i}'].description[:-1]] = globals[f'net_type_y_{i}'].value
                if ohe[globals[f'y_name_{i}'].description[:-1]]:
                    y_shape[globals[f'y_name_{i}'].description[:-1]] = 'Без изменений'

            if checkbox_split.value:
                split_size = [slider.value, int(round(slider2.value / 2, 0)),
                              int(100 - slider.value - round(slider2.value / 2, 0))]
            else:
                split_size = None

            self.prepare_custom_dataset(dic_of_X, dic_of_Y, x_scaler=x_scaler, y_scaler=y_scaler, x_shape=x_shape,
                                y_shape=y_shape, one_hot=ohe, split=split_size)

            if checkbox_google.value:
                directory = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'TerraAI', 'datasets')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(f"{os.path.join(directory, self.name)}.trds", "wb") as f:
                    dill.dump(self, f)
                tzinfo = timezone('Europe/Moscow')
                now = datetime.now().astimezone(tzinfo)
                dt_string = now.isoformat()
                data = {}
                data['name'] = self.name
                data['source'] = self.source
                data['tags'] = list(self.tags.values())
                data['date'] = dt_string
                data['size'] = self._get_size(f'{directory}/{self.name}.trds')
                with open(f'{directory}/{self.name}.trds.json', 'w') as fp:
                    json.dump(data, fp)
                print(f'Датасет сохранен в файл {directory}/{self.name}.trds')
                print(f'Json сохранен в файл {directory}/{self.name}.trds.json')

            pass

        # Первая вкладка
        list_of_rows = []
        disabled_list = []
        for i in range(inputs):
            list_of_widgets = []
            globals[f'input_{i}'] = {}
            globals[f'x_name_{i}'] = widgets.Text(value='', placeholder='Название входа',
                                                  description=f'input_{i + 1}:', disabled=False)
            list_of_widgets.append(globals[f'x_name_{i}'])
            globals[f'x_tag_{i}'] = widgets.Dropdown(
                options=[('Картинки', 'images'), ('Видео', 'video'), ('Текст', 'text'), ('Аудио', 'audio'), ('Временной ряд', 'timeseries'), ('Датафрейм', 'regression'), ('Другое', 'other')], value='other',
                description=f'Тип данных:', disabled=False)
            list_of_widgets.append(globals[f'x_tag_{i}'])
            globals[f'xtrain_{i}'] = widgets.Text(value='x_train', description='X/Train:',
                                                  placeholder='X или x_train', disabled=False)
            list_of_widgets.append(globals[f'xtrain_{i}'])
            globals[f'xval_{i}'] = widgets.Text(value='x_val', description='Validation:', placeholder='',
                                                disabled=False)
            list_of_widgets.append(globals[f'xval_{i}'])
            disabled_list.append(globals[f'xval_{i}'])
            globals[f'xtest_{i}'] = widgets.Text(value='x_test', description='Test:', placeholder='', disabled=False)
            list_of_widgets.append(globals[f'xtest_{i}'])
            disabled_list.append(globals[f'xtest_{i}'])
            globals[f'x_block_{i}'] = widgets.HBox(list_of_widgets)
            list_of_rows.append(globals[f'x_block_{i}'])
            globals[f'input_{i}'][globals[f'x_name_{i}']] = (
                globals[f'xtrain_{i}'], globals[f'xval_{i}'], globals[f'xtest_{i}'])
        for i in range(outputs):
            list_of_widgets = []
            globals[f'output_{i}'] = {}
            globals[f'y_name_{i}'] = widgets.Text(value='', placeholder='Название выхода',
                                                  description=f'output_{i + 1}:', disabled=False)
            list_of_widgets.append(globals[f'y_name_{i}'])
            globals[f'y_tag_{i}'] = widgets.Dropdown(
                options=[('Картинки', 'images'), ('Текст', 'text'), ('Аудио', 'audio'), ('Другое', 'other')],
                value='other', description=f'Тип данных:', disabled=False)
            globals[f'y_task_{i}'] = widgets.Dropdown(
                options=[('Классификация', 'classification'), ('Сегментация', 'segmentation'),
                         ('Обнаружение объектов', 'object_detection'), ('Автокодировщик', 'autoencoder'),
                         ('Генеративно-состязательная сеть', 'gan'), ('Регрессия', 'regression'),
                         ('Временные ряды', 'timeseries'), ('Предсказание временного ряда', 'timeseries_prediction')],
                value='classification',
                description='Тип задачи:',
            )
            globals[f'y_tag_task_{i}'] = widgets.VBox([globals[f'y_tag_{i}'], globals[f'y_task_{i}']])
            list_of_widgets.append(globals[f'y_tag_task_{i}'])
            globals[f'ytrain_{i}'] = widgets.Text(value='y_train', description='Y/Train:',
                                                  placeholder='Y или y_train', disabled=False)
            list_of_widgets.append(globals[f'ytrain_{i}'])
            globals[f'yval_{i}'] = widgets.Text(value='y_val', description='Validation:', placeholder='',
                                                disabled=False)
            list_of_widgets.append(globals[f'yval_{i}'])
            disabled_list.append(globals[f'yval_{i}'])
            globals[f'ytest_{i}'] = widgets.Text(value='y_test', description='Test:', placeholder='', disabled=False)
            list_of_widgets.append(globals[f'ytest_{i}'])
            disabled_list.append(globals[f'ytest_{i}'])
            globals[f'y_block_{i}'] = widgets.HBox(list_of_widgets)
            list_of_rows.append(globals[f'y_block_{i}'])
            globals[f'output_{i}'][globals[f'y_name_{i}']] = (
                globals[f'ytrain_{i}'], globals[f'yval_{i}'], globals[f'ytest_{i}'])

        checkbox_google = widgets.Checkbox(value=True, description='Сохранить в Google Drive', disabled=False)
        checkbox_split = widgets.Checkbox(value=False, description='Train/Val/Test split', disabled=False)

        dataset_name = widgets.Text(value='', description='Датасет:', placeholder='Название датасета', disabled=False)
        dataset_hashtags = widgets.Text(value='', description='Пользовательские теги', placeholder='Теги через пробел', disabled=False)
        button = widgets.Button(description='Сформировать', disabled=False, button_style='', icon='check')
        button.on_click(send_arrays)
        dump_button = widgets.HBox([dataset_name, dataset_hashtags, button])
        slider = widgets.IntSlider(description='Train:', value=80, step=1, min=5, max=95)
        slider2 = widgets.IntSlider(description='Val+test:', value=20, step=1, min=5, max=95)

        def value_changed(change):
            slider2.value = 100 - change.new

        slider.observe(value_changed, 'value')

        def value_changed_2(change):
            slider.value = 100 - change.new

        slider2.observe(value_changed_2, 'value')
        train_list = widgets.HBox([checkbox_google, checkbox_split, slider, slider2])

        for wid in disabled_list:
            widgets.link((checkbox_split, 'value'), (wid, 'disabled'))

        list_of_rows.append(train_list)
        list_of_rows.append(dump_button)
        first_page = widgets.VBox(list_of_rows)

        # Вторая вкладка
        list_of_rows_2 = []
        for i in range(inputs):
            list_of_widgets_2 = []
            globals[f'scaler_x_{i}'] = widgets.Dropdown(options=['Не применять', 'StandardScaler', 'MinMaxScaler'],
                                                        value='Не применять', description=f'x_Scaler_{i + 1}:',
                                                        disabled=False)
            list_of_widgets_2.append(globals[f'scaler_x_{i}'])
            globals[f'net_type_x_{i}'] = widgets.Dropdown(
                options=['Без изменений', 'Добавить размерность', 'Выпрямить'], value='Без изменений',
                description='Размерность:', disabled=False)
            list_of_widgets_2.append(globals[f'net_type_x_{i}'])
            globals[f'block_{i}'] = widgets.HBox(list_of_widgets_2)
            list_of_rows_2.append(globals[f'block_{i}'])
        for i in range(outputs):
            list_of_widgets_2 = []
            globals[f'scaler_y_{i}'] = widgets.Dropdown(options=['Не применять', 'StandardScaler', 'MinMaxScaler'],
                                                        value='Не применять', description=f'y_Scaler_{i + 1}:',
                                                        disabled=False)
            list_of_widgets_2.append(globals[f'scaler_y_{i}'])
            globals[f'net_type_y_{i}'] = widgets.Dropdown(
                options=['Без изменений', 'Добавить размерность', 'Выпрямить'], value='Без изменений',
                description='Размерность:', disabled=False)
            list_of_widgets_2.append(globals[f'net_type_y_{i}'])
            globals[f'ohe_{i}'] = widgets.Checkbox(value=False, description='One-Hot Encoding', disabled=False)
            globals[f'block_{i}'] = widgets.HBox([*list_of_widgets_2, globals[f'ohe_{i}']])
            list_of_rows_2.append(globals[f'block_{i}'])
            for wid in list_of_widgets_2:
                widgets.link((globals[f'ohe_{i}'], 'value'), (wid, 'disabled'))
        list_of_rows_2.append(train_list)
        list_of_rows_2.append(dump_button)
        second_page = widgets.VBox(list_of_rows_2)

        # Соединяем две вкладки
        load_widget = widgets.Tab()
        load_widget.children = [first_page, second_page]
        load_widget.set_title(title='Массивы', index=0)
        load_widget.set_title(title='Предобработка', index=1)

        display(load_widget)

        pass

    def _set_tag(self, name: str) -> list:

        tags = {'mnist': ['images', 'classification'],
                'fashion_mnist': ['images', 'classification'],
                'cifar10': ['images', 'classification'],
                'cifar100': ['images', 'classification'],
                'imdb': ['text', 'classification'],
                'boston_housing': ['text', 'regression'],
                'reuters': ['text', 'classification'],
                'sber': ['timeseries', 'regression'],
                'автомобили': ['images', 'classification'],
                'автомобили_3': ['images', 'classification'],
                'самолеты': ['images', 'segmentation', 'objectDetection'],
                'умный_дом': ['audio', 'classification', 'smartHome'],
                'договоры': ['text', 'segmentation'],
                'трейдинг': ['traiding', 'timeseries'],
                'квартиры': ['text', 'regression'],
                'болезни': ['images', 'segmentation'],
                'заболевания': ['text', 'classification'],
                'губы': ['images', 'segmentation'],
                'жанры_музыки': ['audio', 'classification']
                }
        return tags[name]

    def _set_language(self, name: str):

        language = {'imdb': 'English',
                    'boston_housing': 'English',
                    'reuters': 'English',
                    'заболевания': 'Russian',
                    'договоры': 'Russian',
                    'умный_дом': 'Russian',
                    'квартиры': 'Russian'
                    }

        if name in language.keys():
            return language[name]
        else:
            return None

    def _set_source(self, name: str) -> str:

        source = {'mnist': 'tensorflow.keras',
                  'fashion_mnist': 'tensorflow.keras',
                  'cifar10': 'tensorflow.keras',
                  'cifar100': 'tensorflow.keras',
                  'imdb': 'tensorflow.keras',
                  'boston_housing': 'tensorflow.keras',
                  'reuters': 'tensorflow.keras',
                  'sber': 'Terra AI',
                  'автомобили': 'Terra AI',
                  'автомобили_3': 'Terra AI',
                  'самолеты': 'Terra AI',
                  'умный_дом': 'Terra AI',
                  'договоры': 'Terra AI',
                  'трейдинг': 'Terra AI',
                  'квартиры': 'Terra AI',
                  'болезни': 'Terra AI',
                  'заболевания': 'Terra AI',
                  'губы': 'Terra AI',
                  'жанры_музыки': 'Terra AI'
                  }

        if name in source.keys():
            return source[name]
        else:
            return 'custom dataset'

    def _set_datatype(self, **kwargs) -> str:

        dtype = {0: 'DIM',
                 1: 'DIM',
                 2: 'DIM',
                 3: '1D',
                 4: '2D',
                 5: '3D'
                 }

        if 'shape' in kwargs.keys():
            return dtype[len(kwargs['shape'])]
        elif 'text' in kwargs.keys() and kwargs['text'] == True:
            return 'Text'

    def _get_zipfiles(self) -> list:
        from django.conf import settings
        # return os.listdir('/content/drive/MyDrive/TerraAI/datasets/sources')
        return os.listdir(os.path.join(settings.TERRA_AI_DATA_PATH, 'datasets', 'sources'))

    def _get_size(self, path) -> str:

        size_bytes = os.path.getsize(path)

        con = 1024
        if 0 <= size_bytes <= con:
            size = f'{size_bytes} B'
        elif con <= size_bytes <= con ** 2:
            size = f'{round(size_bytes / con, 2)} KB'
        elif con ** 2 <= size_bytes <= con ** 3:
            size = f'{round(size_bytes / con ** 2, 2)} MB'
        elif con ** 3 <= size_bytes <= con ** 4:
            size = f'{round(size_bytes / con ** 3, 2)} GB'

        return size

    def _find_colors(self, name: str, num_classes=None, mask_range=None, txt_file=False) -> list:

        if txt_file:
            txt = pd.read_csv(os.path.join(self.file_folder, name), sep=':')
            color_list = {}
            for i in range(len(txt)):
                color_list[txt.loc[i, '# label']] = [int(num) for num in txt.loc[i, 'color_rgb'].split(',')]
        else:
            color_list = []

            for img in sorted(os.listdir(os.path.join(self.file_folder, name))):
                path = os.path.join(self.file_folder, name, img)
                width, height = Image.open(path).size
                img = load_img(path, target_size=(height, width))
                array = img_to_array(img).astype('uint8')

                image = array.reshape(-1, 3)
                km = KMeans(n_clusters=num_classes)
                km.fit(image)
                labels = km.labels_
                cl_cent = np.round(km.cluster_centers_).astype('uint8')[:max(labels) + 1].tolist()
                for color in cl_cent:
                    if color_list:
                        if color not in color_list:
                            for in_color in color_list:
                                if color[0] in range(in_color[0] - mask_range, in_color[0] + mask_range) and color[
                                    1] in range(in_color[1] - mask_range, in_color[1] + mask_range) and color[2] in range(
                                        in_color[2] - mask_range, in_color[2] + mask_range):
                                    add_condition = False
                                    break
                                else:
                                    add_condition = True
                            if add_condition:
                                color_list.append(color)
                    else:
                        color_list.append(color)
                if len(color_list) >= num_classes:
                    break

        return color_list

    def load_data(self, name: str, mode: str, link=None):

        """
        Create folder and download base in it. Does not change the files original format.
        If base is on the standard base list also print detailed information about specified base.

        Examples:
            trds.DataLoader().load_data('договоры');

            trds.DataLoader().load_data('base_name', url)
        Args:
            name (str): name of the base for downloading;

            link (str): url where base is located
        """

        data = {
            # 'трафик': ['traff.csv'],
            'трейдинг': ['trading.zip'],
            'автомобили': ['cars.zip'],
            'умный_дом': ['smart_home.zip'],
            'квартиры': ['flats.zip'],
            # 'диалоги': ['dialog.txt'],
            'автомобили_3': ['cars_3.zip'],
            'заболевания': ['diseases.zip'],
            'договоры': ['docs.zip'],
            'самолеты': ['planes.zip'],
            # 'болезни': ['origin.zip', 'segmentation.zip'],
            'губы': ['lips.zip'],
            # 'жанры_музыки': ['genres.zip'],
            'sber': ['sber.zip']
        }

        reference = {
            'трафик': 'Файл в формате csv, содержит информацию о трафике на сайт компании в виде таблицы из двух '
                      'столбцов.\nПервый столбец - дата, второй столбец - объем трафика за сутки.\nВ таблице отражены '
                      'данные за 1095 дней\nСсылка на базу: https://storage.googleapis.com/terra_ai/DataSets/traff.csv',
            'трейдинг': 'Содержит 7 файлов в формате txt с информацией по котировкам акций 3 компаний.\nДанные '
                        'представлены в виде таблицы, отражающей показатели стоимости и объема торгов в '
                        'динамике.\nВсего 9 показателей (компания, дата, время, цена открытия и т.д.)\nСсылка на '
                        'базу: https://storage.googleapis.com/terra_ai/DataSets/shares.zip',
            'автомобили': 'Содержит две папки с изображениями автомобилей в формате png.\nВ первой папке 1088 '
                          'фотографий автомобилей марки Феррари, во второй папке 1161 фотография автомобилей марки '
                          'Мерседес\nСсылка на базу: https://storage.googleapis.com/terra_ai/DataSets/car_2.zip',
            'умный_дом': 'База состоит из звуковых файлов в формате wav для обучения системы "умный дом".\nБаза '
                         'разделена на тренировочную и тестовую выбороки.\nТренировочная выборка включает 4 папки, '
                         '3 из которых содержат  по 104 файла записью одного из типов команд:\nкондиционер, свет, '
                         'телевизор. И 1 папку с записью обычной речи - 50 файлов.\nТестовая выборка содержит запись '
                         'фрагментов речи - 50 файлов \nСсылка на базу: '
                         'https://storage.googleapis.com/terra_ai/DataSets/cHome.zip',
            'квартиры': 'Файл в формате csv, содержит информацию по квартирам, выставленным на продажу.\nДанные '
                        'представлены в виде таблицы из 14 столбцов, отражающих параметры квартиры (станция метро, '
                        'площадь, тип дома, цена и т.д.)\nВ таблице отражены данные по 252 536 квартирам.',
            'диалоги': 'Файл в формате txt, содержит текстовые данные в виде набора блоков "вопрос-ответ".\nКаждый '
                       'блок занимает 2 строки. Всего в файле около 76 тысяч блоков.\nСсылка на базу: '
                       'https://storage.googleapis.com/terra_ai/DataSets/dialog.txt',
            'автомобили_3': 'Содержит три папки с изображениями автомобилей в формате png.\nВ первой папке 1088 '
                            'фотографий Феррари, во второй папке 1161 фотография автомобилей марки Мерседес\nВ третье '
                            'папке 1178 фотографий автомобилей марки Рено.\nСсылка на базу: '
                            'https://storage.googleapis.com/terra_ai/DataSets/car.zip',
            'заболевания': 'База содержит 10 папок с файлами в формате txt с описаниями симптомов 10 заболеваний '
                           'желудочно-кишечного тракта\nСсылка на '
                           'базу:https://storage.googleapis.com/terra_ai/DataSets/symptoms.zip',
            # 'люди':['master.zip', 'coco2017val.zip', 'coco128.zip'], # Этой базы нет в aiu_bucket
            'договоры': 'Содержит 428 файлов в формате txt с текстами договоров.\nСсылка на базу: '
                        'https://storage.googleapis.com/terra_ai/DataSets/docs.zip',
            'самолеты': 'Содержит две папки с изображениями самолетов в формате jpg.\nВ папке "Самолеты" размещен 981 '
                        'файл с исходными изображениями.\nВ папке "Сегменты" размещен 981 файл с сегментированными '
                        'изображениями.\nСсылка на базу: исходные изображения '
                        'https://storage.googleapis.com/terra_ai/DataSets/airplane.zip'
                        ',\n                сегментированные изобажения '
                        'https://storage.googleapis.com/terra_ai/DataSets/segment.zip',
            'болезни': 'Содержит две папки с файлами в формате jpg. \nВ папке ''origin''размещены исходные  '
                       'фотографии, с примерами поражения кожи при 10 дерматологических заболеваниях:\nПсориаз 500 '
                       '\nДерматит 500 \nГерпес 499 \nАкне 510 \nНевус 495 \nВитилиго 504 \nХлоазма 499 \nЭкзема 498 '
                       '\nСыпь 522 \nЛишай 502\nВ папке ''segmentation'' размещены сегментированные '
                       'изображения.\nСсылка на базу: исходные изображения '
                       'https://storage.googleapis.com/terra_ai/DataSets/origin.zip \n                сегментированные '
                       'изображения https://storage.googleapis.com/terra_ai/DataSets/segmentation.zip'
        }

        default_path = self.save_path
        if mode == 'google_drive':
            filepath = os.path.join('/content/drive/MyDrive/TerraAI/datasets/sources', name)
            name = name[:name.rfind('.')]
            file_folder = os.path.join(default_path, name)
            shutil.unpack_archive(filepath, file_folder)
        elif mode == 'url':
            filename = link.split('/')[-1]
            file_folder = pathlib.Path(os.path.join(default_path, filename))
            if '.' in filename:
                name = filename[:filename.rfind('.')]
                file_folder = pathlib.Path(os.path.join(default_path, name))
            if not file_folder.exists():
                os.makedirs(file_folder)
            if not file_folder.joinpath('tmp').exists():
                os.makedirs(os.path.join(file_folder, 'tmp'))
            with request.urlopen(link) as dl_file:
                with open(os.path.join(file_folder, 'tmp', filename), 'wb') as out_file:
                    out_file.write(dl_file.read())
            if 'zip' in filename or 'zip' in link:
                file_path = pathlib.Path(os.path.join(file_folder, 'tmp', filename))
                temp_folder = os.path.join(file_folder, 'tmp')
                shutil.unpack_archive(file_path, file_folder)
                shutil.rmtree(temp_folder, ignore_errors=True)
        elif mode == 'terra':
            if name in data.keys():
                self.language = self._set_language(name)
                for base in data[name]:
                    file_folder = pathlib.Path(default_path).joinpath(name)
                    if not file_folder.exists():
                        os.makedirs(file_folder)
                    if not file_folder.joinpath('tmp').exists():
                        os.makedirs(os.path.join(file_folder, 'tmp'))
                    link = 'https://storage.googleapis.com/terra_ai/DataSets/Numpy/' + base
                    with request.urlopen(link) as dl_file:
                        with open(os.path.join(default_path, name, 'tmp', base), 'wb') as out_file:
                            out_file.write(dl_file.read())
                    if 'zip' in base:
                        file_path = pathlib.Path(os.path.join(default_path, name, 'tmp', base))
                        temp_folder = file_folder.joinpath('tmp')
                        if not temp_folder.exists:
                            os.mkdir(temp_folder)
                        shutil.unpack_archive(file_path, file_folder)
                        shutil.rmtree(temp_folder, ignore_errors=True)
                if not self.django_flag:
                    if name in reference.keys():
                        print(reference[name])
            else:
                if not name in data.keys():
                    if self.django_flag:
                        self.Exch.print_error(('Error', 'Данной базы нет в списке готовых баз.'))
                    else:
                        assert name in data.keys(), 'Данной базы нет в списке готовых баз.'
        self.name = name
        self.file_folder = str(file_folder)
        self.source = self._set_source(name)
        if not self.django_flag:
            print(f'Файлы скачаны в директорию {self.file_folder}')

        return self

    def prepare_dataset(self, **options):

        def load_arrays():

            inp_datatype = []
            for arr in os.listdir(os.path.join(self.file_folder, 'arrays')):
                if arr[0] == 'X':
                    self.X[arr[2:-3]] = {'data_name': f'Вход_{arr[-4]}',
                                         'data': joblib.load(os.path.join(self.file_folder, 'arrays', arr))}
                    self.input_shape[arr[2:-3]] = self.X[arr[2:-3]]['data'][0].shape[1:]
                    inp_datatype.append(self._set_datatype(shape=self.X[arr[2:-3]]['data'][0].shape))
                    self.tags[arr[2:-3]] = tag_list[0]
                elif arr[0] == 'Y':
                    self.Y[arr[2:-3]] = {'data_name': f'Выход_{arr[-4]}',
                                         'data': joblib.load(os.path.join(self.file_folder, 'arrays', arr))}
                    self.output_shape[arr[2:-3]] = self.Y[arr[2:-3]]['data'][0].shape[1:]
                    self.output_datatype[arr[2:-3]] = self._set_datatype(shape=self.Y[arr[2:-3]]['data'][0].shape)
                    self.tags[arr[2:-3]] = tag_list[1]
            self.input_datatype = ' '.join(inp_datatype)

        def load_scalers():

            if 'scalers' in os.listdir(self.file_folder):
                X_scalers = [sclr[2:-3] for sclr in os.listdir(os.path.join(self.file_folder, 'scalers')) if 'X' in sclr]
                Y_scalers = [sclr[2:-3] for sclr in os.listdir(os.path.join(self.file_folder, 'scalers')) if 'Y' in sclr]
            else:
                X_scalers = []
                Y_scalers = []

            for inp in list_of_inputs:
                if inp in X_scalers:
                    self.x_Scaler[inp] = joblib.load(os.path.join(self.file_folder, 'scalers', f'X_{inp}.gz'))
                else:
                    self.x_Scaler[inp] = None

            for out in list_of_outputs:
                if out in Y_scalers:
                    self.y_Scaler[out] = joblib.load(os.path.join(self.file_folder, 'scalers', f'Y_{out}.gz'))
                else:
                    self.y_Scaler[out] = None

            pass

        def load_tokenizer():

            if 'tokenizer' in os.listdir(self.file_folder):
                tokenizer = [tok[0:-3] for tok in os.listdir(os.path.join(self.file_folder, 'tokenizer'))]
            else:
                tokenizer = []

            for inp in list_of_inputs:
                if inp in tokenizer:
                    self.tokenizer[inp] = joblib.load(os.path.join(self.file_folder, 'tokenizer', f'{inp}.gz'))
                else:
                    self.tokenizer[inp] = None

            pass

        def load_word2vec():

            if 'word2vec' in os.listdir(self.file_folder):
                word2v = [w2v[0:-3] for w2v in os.listdir(os.path.join(self.file_folder, 'word2vec'))]
            else:
                word2v = []

            for inp in list_of_inputs:
                if inp in word2v:
                    self.word2vec[inp] = joblib.load(os.path.join(self.file_folder, 'word2vec', f'{inp}.gz'))
                else:
                    self.word2vec[inp] = None

            pass

        if options['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters']:

            if options['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, scaler='MinMaxScaler', net='conv', test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'imdb':
                self.keras_datasets(options['dataset_name'], one_hot_encoding=True, test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'reuters':
                self.keras_datasets(options['dataset_name'], test=True)
                self.task_type['output_1'] = 'classification'
            elif options['dataset_name'] == 'boston_housing':
                self.keras_datasets(options['dataset_name'], scaler='StandardScaler', test=True)
                self.task_type['output_1'] = 'regression'

        else:

            self.load_data(options['dataset_name'], mode='terra')

            config = configparser.ConfigParser()
            config.read(os.path.join(self.file_folder, 'config.ini'), encoding="utf-8")
            name = config.get('ATTRIBUTES', 'name')
            if name == 'Акции сбербанка':
                name = 'sber'
            elif name == 'Акции газпрома':
                name = 'трейдинг'
            num_classes = literal_eval(config.get('ATTRIBUTES', 'num_classes'))
            if isinstance(num_classes, dict): #ЗАПЛАТКА из за базы "трейдинг" и "сбер"
                num_classes = num_classes['output_1']
            self.name = name
            self.source_datatype = list(literal_eval(config.get('ATTRIBUTES', 'source_datatype')).values())[0]
            self.source_shape = literal_eval(config.get('ATTRIBUTES', 'source_shape'))
            self.classes_names['output_1'] = literal_eval(config.get('ATTRIBUTES', 'classes_names'))
            self.classes_colors['output_1'] = literal_eval(config.get('ATTRIBUTES', 'classes_colors'))
            self.num_classes['output_1'] = num_classes
            self.task_type = literal_eval(config.get('ATTRIBUTES', 'task_type'))
            self.one_hot_encoding = literal_eval(config.get('ATTRIBUTES', 'one_hot_encoding'))
            if self.name == 'договоры': # TODO - ЗАПЛАТКА!!!!!!
                self.task_type['output_1'] = 'segmentation'
            tag_list = self._set_tag(self.name)

            # folder_list = sorted([X for X in os.listdir(self.file_folder) if os.path.isdir(os.path.join(self.file_folder, X))])
            folder_list = ['arrays', 'scalers', 'tokenizer', 'word2vec']
            progress_bar = tqdm(folder_list, ncols=800)
            idx = 0
            for i, folder in enumerate(progress_bar):
                progress_bar.set_description('Загрузка файлов')
                if folder == 'arrays':
                    load_arrays()
                    list_of_inputs = self.X.keys()
                    list_of_outputs = self.Y.keys()
                elif folder == 'scalers':
                    load_scalers()
                elif folder == 'tokenizer':
                    load_tokenizer()
                elif folder == 'word2vec':
                    load_word2vec()
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total:
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)
            self.dts_prepared = True

        return self

    def prepare_custom_dataset(self, *Data, **options):

        inputs = list(Data[0].keys())
        outputs = list(Data[1].keys())
        self.X = Data[0]
        self.Y = Data[1]
        if options['split']:
            indices = np.random.permutation(Data[0][inputs[0]]['data'][0].shape[0])
            train_len = int(options['split'][0] / 100 * len(indices))
            val_len = int((len(indices) - train_len) / 2)
            train_mask = indices[:train_len]
            val_mask = indices[train_len:train_len + val_len]
            test_mask = indices[train_len + val_len:]
            for inp in inputs:
                self.X[inp]['data'] = (Data[0][inp]['data'][0][train_mask], Data[0][inp]['data'][0][val_mask],
                                       Data[0][inp]['data'][0][test_mask])
            for out in outputs:
                self.Y[out]['data'] = (Data[1][out]['data'][0][train_mask], Data[1][out]['data'][0][val_mask],
                                       Data[1][out]['data'][0][test_mask])

        self.source = 'custom dataset'
        source_datatype = []
        inp_datatype = []

        for inp in inputs:

            self.source_shape[inp] = self.X[inp]['data'][0].shape[1:]
            source_datatype.append(self._set_datatype(shape=self.X[inp]['data'][0].shape))
            if options['x_scaler'][inp] in ['StandardScaler', 'MinMaxScaler']:
                if options['x_scaler'][inp] == 'MinMaxScaler':
                    self.x_Scaler[inp] = MinMaxScaler()
                elif options['x_scaler'][inp] == 'StandardScaler':
                    self.x_Scaler[inp] = StandardScaler()
                list_of_arrays = []
                self.x_Scaler[inp].fit(self.X[inp]['data'][0].reshape(-1, 1))
                for array in self.X[inp]['data']:
                    if isinstance(array, np.ndarray):
                        shape_x = array.shape
                        array = self.x_Scaler[inp].transform(array.reshape(-1, 1))
                        array = array.reshape(shape_x)
                    else:
                        array = None
                    list_of_arrays.append(array)
                self.X[inp]['data'] = tuple(list_of_arrays)
                del list_of_arrays
            else:
                self.x_Scaler[inp] = None

            if options['x_shape'][inp] in ['Добавить размерность', 'Выпрямить']:
                if options['x_shape'][inp] == 'Добавить размерность':
                    list_of_arrays = []
                    for array in self.X[inp]['data']:
                        if isinstance(array, np.ndarray):
                            list_of_arrays.append(array[..., None])
                        else:
                            list_of_arrays.append(None)
                    self.X[inp]['data'] = tuple(list_of_arrays)
                    del list_of_arrays
                elif options['x_shape'][inp] == 'Выпрямить':
                    list_of_arrays = []
                    for array in self.X[inp]['data']:
                        if isinstance(array, np.ndarray):
                            list_of_arrays.append(array.reshape(-1, np.prod(array.shape[1:])))
                        else:
                            list_of_arrays.append(None)
                    self.X[inp]['data'] = tuple(list_of_arrays)
                    del list_of_arrays

            self.input_shape[inp] = self.X[inp]['data'][0].shape[1:]
            inp_datatype.append(self._set_datatype(shape=self.X[inp]['data'][0].shape))

        for out in outputs:
            if options['y_scaler'][out] in ['StandardScaler', 'MinMaxScaler']:
                if options['y_scaler'][out] == 'MinMaxScaler':
                    self.y_Scaler[out] = MinMaxScaler()
                elif options['y_scaler'][out] == 'StandardScaler':
                    self.y_Scaler[out] = StandardScaler()
                list_of_arrays = []
                self.y_Scaler[out].fit(self.Y[out]['data'][0].reshape(-1, 1))
                for array in self.Y[out]['data']:
                    if isinstance(array, np.ndarray):
                        shape_y = array.shape
                        array = self.y_Scaler[out].transform(array.reshape(-1, 1))
                        array = array.reshape(shape_y)
                    else:
                        array = None
                    list_of_arrays.append(array)
                self.Y[out]['data'] = tuple(list_of_arrays)
                del list_of_arrays
            else:
                self.y_Scaler[out] = None

            if options['y_shape'][out] in ['Добавить размерность', 'Выпрямить']:
                if options['y_shape'][out] == 'Добавить размерность':
                    list_of_arrays = []
                    for array in self.Y[out]['data']:
                        if isinstance(array, np.ndarray):
                            list_of_arrays.append(array[..., None])
                        else:
                            list_of_arrays.append(None)
                    self.Y[out]['data'] = tuple(list_of_arrays)
                    del list_of_arrays
                elif options['y_shape'][out] == 'Выпрямить':
                    list_of_arrays = []
                    for array in self.Y[out]['data']:
                        if isinstance(array, np.ndarray):
                            list_of_arrays.append(array.reshape(-1, np.prod(array.shape[1:])))
                        else:
                            list_of_arrays.append(None)
                    self.Y[out]['data'] = tuple(list_of_arrays)
                    del list_of_arrays

            if options['one_hot'][out]:
                list_of_arrays = []
                for array in self.Y[out]['data']:
                    if isinstance(array, np.ndarray):
                        arr = utils.to_categorical(array, len(np.unique(array, axis=0)))
                    else:
                        arr = None
                    list_of_arrays.append(arr)
                self.Y[out]['data'] = tuple(list_of_arrays)
                self.one_hot_encoding[out] = True
                del list_of_arrays
            else:
                self.one_hot_encoding[out] = False
            self.output_shape[out] = self.Y[out]['data'][0].shape[1:]
            self.output_datatype[out] = self._set_datatype(shape=self.Y[out]['data'][0].shape)

        self.source_datatype = ' '.join(source_datatype)
        self.input_datatype = ' '.join(inp_datatype)
        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено.')
            for inp in inputs:
                x_arrays = ['Train', 'Validation', 'Test']
                for i, item_x in enumerate(self.X[inp]['data']):
                    if item_x is not None:
                        print(f"{inp} {x_arrays[i]}: {item_x.shape}")
            for out in outputs:
                y_arrays = ['Train', 'Validation', 'Test']
                for i, item_y in enumerate(self.Y[out]['data']):
                    if item_y is not None:
                        print(f"{out} {y_arrays[i]}: {item_y.shape}")

        return self

    def inverse_data(self, array=None, scaler=None):

        #Не доделано
        if scaler:
            array = self.__dict__[scaler].inverse_transform(array)

        return array

    def keras_datasets(self, dataset: str, **options):

        def print_data(name, x_train, y_train):

            pics = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
            text = ['imdb', 'reuters', 'boston_housing']

            if name in pics:
                fig, axs = plt.subplots(1, 10, figsize=(25, 3))
                for i in range(10):
                    label_indexes = np.where(y_train == i)[0]
                    index = random.choice(label_indexes)
                    img = x_train[index]
                    title = y_train[index]
                    if name in ['mnist', 'fashion_mnist']:
                        axs[i].imshow(Image.fromarray(img), cmap='gray')
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names["output_1"][title]}')
                    else:
                        axs[i].imshow(Image.fromarray(img))
                        axs[i].axis('off')
                        axs[i].set_title(f'{i}: {self.classes_names["output_1"][title[0]]}')

            if name in text:
                if name in ['imdb', 'reuters']:
                    pd.DataFrame({'x_train': x_train, 'y_train': y_train}).head()
                else:
                    df = pd.DataFrame(x_train)
                    df['y_train'] = y_train
                    df.head()

            pass

        cur_time = time()
        self.name = dataset.lower()
        tags = self._set_tag(self.name)
        self.tags = {'input_1': tags[0],
                     'output_1': tags[1]}
        self.source = 'tensorflow.keras'
        data = {
            'mnist': mnist,
            'fashion_mnist': fashion_mnist,
            'cifar10': cifar10,
            'cifar100': cifar100,
            'imdb': imdb,
            'reuters': reuters,
            'boston_housing': boston_housing
        }
        if not dataset.lower() in data.keys():
            self.Exch.print_error(('Error', 'Данного датасета нет в списке стандартных датасетов keras.'))
            if not self.django_flag:
                assert dataset.lower() in data.keys(), 'Данного датасета нет в списке стандартных датасетов keras.'
        progress_bar = tqdm(range(1), ncols=800)
        progress_bar.set_description(f'Загрузка датасета {self.name}')
        idx = 0
        for _ in progress_bar:
            (x_Train, y_Train), (x_Val, y_Val) = data[self.name].load_data()
            if self.django_flag:
                idx += 1
                progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                       f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)

        self.source_shape['input_1'] = x_Train.shape if len(x_Train.shape) < 2 else x_Train.shape[1:]
        self.language = self._set_language(self.name)
        self.source_datatype += f' {self._set_datatype(shape=x_Train.shape)}'
        if 'classification' in self.tags['output_1']:
            self.num_classes['output_1'] = len(np.unique(y_Train, axis=0))
            if self.name == 'fashion_mnist':
                self.classes_names['output_1'] = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                      'Sneaker', 'Bag', 'Ankle boot']
            elif self.name == 'cifar10':
                self.classes_names['output_1'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                      'truck']
            else:
                self.classes_names['output_1'] = [str(i) for i in range(len(np.unique(y_Train, axis=0)))]
        else:
            self.num_classes['output_1'] = 1

        if not self.django_flag:
            print_data(self.name, x_Train, y_Train)

        if 'net' in options.keys() and self.name in list(data.keys())[:4]:
            if options['net'].lower() == 'linear':
                x_Train = x_Train.reshape((-1, np.prod(np.array(x_Train.shape)[1:])))
                x_Val = x_Val.reshape((-1, np.prod(np.array(x_Val.shape)[1:])))
            elif options['net'].lower() == 'conv':
                if len(x_Train.shape) == 3:
                    x_Train = x_Train[..., None]
                    x_Val = x_Val[..., None]

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

            if self.name == 'imdb' or self.name == 'reuters':
                if not self.django_flag:
                    print(f'Scaling required dataset is currently unavaliable. {options["scaler"]} was not implemented.')
            else:
                shape_xt = x_Train.shape
                shape_xv = x_Val.shape
                x_Train = x_Train.reshape(-1, 1)
                x_Val = x_Val.reshape(-1, 1)

                if 'classification' not in self.tags['output_1']:
                    shape_yt = y_Train.shape
                    shape_yv = y_Val.shape
                    y_Train = y_Train.reshape(-1, 1)
                    y_Val = y_Val.reshape(-1, 1)

                self.y_Scaler['output_1'] = None
                if options['scaler'] == 'MinMaxScaler':
                    self.x_Scaler['input_1'] = MinMaxScaler()
                    if 'classification' not in self.tags['output_1']:
                        self.y_Scaler['output_1'] = MinMaxScaler()

                elif options['scaler'] == 'StandardScaler':
                    self.x_Scaler['input_1'] = StandardScaler()
                    if 'classification' not in self.tags['output_1']:
                        self.y_Scaler['output_1'] = StandardScaler()

                self.x_Scaler['input_1'].fit(x_Train)
                x_Train = self.x_Scaler['input_1'].transform(x_Train)
                x_Val = self.x_Scaler['input_1'].transform(x_Val)
                x_Train = x_Train.reshape(shape_xt)
                x_Val = x_Val.reshape(shape_xv)
                if 'classification' not in self.tags['output_1']:
                    self.y_Scaler['output_1'].fit(y_Train)
                    y_Train = self.y_Scaler['output_1'].transform(y_Train)
                    y_Val = self.y_Scaler['output_1'].transform(y_Val)
                    y_Train = y_Train.reshape(shape_yt)
                    y_Val = y_Val.reshape(shape_yv)

        self.one_hot_encoding['output_1'] = False
        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] == True:
            if 'classification' in self.tags['output_1']:
                y_Train = utils.to_categorical(y_Train, len(np.unique(y_Train, axis=0)))
                y_Val = utils.to_categorical(y_Val, len(np.unique(y_Val, axis=0)))
                self.one_hot_encoding['output_1'] = True
            else:
                if not self.django_flag:
                    print(f'One-Hot encoding only available for classification which {self.name} was not meant for. '
                          f'One-Hot encoding was not implemented.')

        self.input_shape['input_1'] = x_Train.shape if len(x_Train.shape) < 2 else x_Train.shape[1:]
        self.input_datatype = self._set_datatype(shape=x_Train.shape)
        self.output_shape['output_1'] = y_Train.shape[1:]
        self.output_datatype['output_1'] = self._set_datatype(shape=y_Train.shape)

        self.X = {'input_1': {'data_name': 'Вход',
                              'data': (x_Train, x_Val, None)}}
        self.Y = {'output_1': {'data_name': 'Выход',
                               'data': (y_Train, y_Val, None)}}

        if 'test' in options.keys() and options['test'] is True:
            split_ratio = self.divide_ratio[1][1:]
            split_size = min(split_ratio) / sum(split_ratio)
            x_Val, x_Test, y_Val, y_Test = train_test_split(x_Val, y_Val, test_size=1 - split_size, shuffle=True)
            self.X['input_1']['data'] = (x_Train, x_Val, x_Test)
            self.Y['output_1']['data'] = (y_Train, y_Val, y_Test)

        self.dts_prepared = True
        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            x_arrays = ['x_train', 'x_val', 'x_test']
            for i, item_x in enumerate(self.X['input_1']['data']):
                if item_x is not None:
                    print(f"Размерность {x_arrays[i]}: {item_x.shape}")
            y_arrays = ['y_train', 'y_val', 'y_test']
            for i, item_y in enumerate(self.Y['output_1']['data']):
                if item_y is not None:
                    print(f"Размерность {y_arrays[i]}: {item_y.shape}")

        return self

    def images(self, folder_name=[''], height=176, width=220, net=['Convolutional', 'Linear'],
               scaler=['No Scaler', 'StandardScaler', 'MinMaxScaler']) -> np.ndarray:

        def load_image(img_path, shape):

            img = load_img(img_path, target_size=shape)
            array = img_to_array(img)

            return array.astype('uint8')

        if folder_name == None:
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]
        shape = (height, width)
        X = []
        Y_cls = []

        for _, dirnames, filename in sorted(os.walk(working_folder)):

            folders = sorted(dirnames)
            folders_num = len(dirnames) if len(dirnames) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except:
                    IndexError

                files = sorted(os.listdir(temp_path))
                for j in range(len(self.user_parameters['out'])):
                    if self.user_parameters['out'][f'output_{j+1}']['tag'] == 'object_detection':

                        data = {}
                        with open(os.path.join(self.file_folder, 'obj.data'), 'r') as dt:
                            d = dt.read()
                        for elem in d.split('\n'):
                            if elem:
                                elem = elem.split(' = ')
                                data[elem[0]] = elem[1]

                        files = []
                        with open(os.path.join(self.file_folder, data["train"].split("/")[-1]), 'r') as dt:
                            imgs = dt.read()
                        for elem in imgs.split('\n'):
                            if elem:
                                files.append(os.path.join(*elem.split('/')[2:]))
                        break

                progress_bar = tqdm(files, ncols=800)
                if folders_num == 1:
                    description = f'Сохранение изображений'
                else:
                    description = f'Сохранение изображений из папки {folders[i]}'
                progress_bar.set_description(description)
                if len(self.source_shape) < 3:
                    source_shape = Image.open(os.path.join(temp_path, os.listdir(temp_path)[0])).size
                    self.source_shape[f'input_{self.iter}'] = (source_shape[1], source_shape[0], 3)
                idx = 0
                for file in progress_bar:
                    X.append(load_image(os.path.join(temp_path, file), shape))
                    Y_cls.append(i)
                    idx += 1
                    if self.django_flag:
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total and i+1 == folders_num:
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)
                self.peg.append(idx+self.peg[-1])

            break

        X = np.array(X)
        Y_cls = np.array(Y_cls)
        self.source_datatype += f' {self._set_datatype(shape=X.shape)}'

        if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
            shape_x = X.shape
            X = X.reshape(-1, 1)
            if scaler == 'MinMaxScaler':
                self.x_Scaler[f'input_{self.iter}'] = MinMaxScaler()
            elif scaler == 'StandardScaler':
                self.x_Scaler[f'input_{self.iter}'] = StandardScaler()
            self.x_Scaler[f'input_{self.iter}'].fit(X)
            X = self.x_Scaler[f'input_{self.iter}'].transform(X)
            X = X.reshape(shape_x)
        else:
            self.x_Scaler[f'input_{self.iter}'] = False

        if net == 'Linear':
            X = X.reshape(-1, np.prod(np.array(X.shape)[1:]))

        if 'classification' in self.task_type.values():
            self.y_Cls = Y_cls.astype('int')
            del Y_cls

        return X

    # def video(self, folder_name=[''], height=64, width=64, max_frames_per_class=10000, scaler=['No Scaler', 'StandardScaler', 'MinMaxScaler']) -> np.ndarray:
    #
    #     if folder_name == None:
    #         folder_name = self.file_folder
    #     else:
    #         folder_name = os.path.join(self.file_folder, folder_name)
    #
    #     X = []
    #     Y_cls = []
    #
    #     for _, dirnames, filename in sorted(os.walk(folder_name)):
    #
    #         folders = sorted(dirnames)
    #         folders_num = len(dirnames) if len(dirnames) != 0 else 1
    #         for i in range(folders_num):
    #             temp_path = folder_name
    #             try:
    #                 temp_path = os.path.join(folder_name, folders[i])
    #             except:
    #                 IndexError
    #
    #             files = sorted(os.listdir(temp_path))
    #             progress_bar = tqdm(files, ncols=800)
    #             if folders_num == 1:
    #                 description = f'Сохранение видеофайлов'
    #             else:
    #                 description = f'Сохранение видеофайлов из папки {folders[i]}'
    #             progress_bar.set_description(description)
    #             idx = 1
    #             for file in progress_bar:
    #                 vc = cv2.VideoCapture(os.path.join(temp_path, file))
    #                 while True:
    #                     success, frame = vc.read()
    #                     if not success:
    #                         break
    #                     resized_frame = cv2.resize(frame, (height, width))
    #                     X.append(resized_frame)
    #                     Y_cls.append(i)
    #                 if self.django_flag:
    #                     idx += 1
    #                     progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
    #                                            f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
    #                     if idx == progress_bar.total and i+1 == folders_num:
    #                         self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
    #                     else:
    #                         self.Exch.print_progress_bar(progress_bar_status)
    #         break
    #
    #     X = np.array(X)
    #     Y_cls = np.array(Y_cls)
    #
    #     if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
    #
    #         shape_x = X.shape
    #         X = X.reshape(-1, 1)
    #
    #         if scaler == 'MinMaxScaler':
    #             self.x_Scaler[f'input_{self.iter}'] = MinMaxScaler()
    #
    #         elif scaler == 'StandardScaler':
    #             self.x_Scaler[f'input_{self.iter}'] = StandardScaler()
    #
    #         self.x_Scaler[f'input_{self.iter}'].fit(X)
    #         X = self.x_Scaler[f'input_{self.iter}'].transform(X)
    #         X = X.reshape(shape_x)
    #
    #     if 'classification' in self.task_type.values():
    #         self.y_Cls = Y_cls.astype('int')
    #         del Y_cls
    #
    #     return X

    def text(self, folder_name=[''], delete_symbols='', x_len=100, step=30, max_words_count=20000, pymorphy=False, bag_of_words=False, embedding=False, embedding_size=200) -> np.ndarray:

        def read_text(file_path):

            del_symbols = ['\n', '\t', '\ufeff']
            if delete_symbols:
                del_symbols += delete_symbols.split(' ')
            with ioopen(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
                for del_symb in del_symbols:
                    text = text.replace(del_symb, ' ')
            for i in range(len(self.user_parameters['out'])):
                if self.user_parameters['out'][f'output_{i+1}']['tag'] == 'text_segmentation':
                    open_symb = self.user_parameters['out'][f'output_{i+1}']['parameters']['open_tags'].split(' ')[0][0]
                    close_symb = self.user_parameters['out'][f'output_{i+1}']['parameters']['open_tags'].split(' ')[0][-1]
                    text = re.sub(open_symb, f" {open_symb}", text)
                    text = re.sub(close_symb, f"{close_symb} ", text)
                    break

            return text

        def apply_pymorphy(text):

            morph = pymorphy2.MorphAnalyzer()
            words = text.split(' ')
            words = [morph.parse(word)[0].normal_form for word in words]

            return words

        def get_set_from_indexes(word_indexes, x_len, step):

            sample = []
            words_len = len(word_indexes)

            index = 0
            peg_idx = 0
            while index + x_len <= words_len:
                sample.append(word_indexes[index:index + x_len])
                index += step
                peg_idx += 1
            if not embedding:
                self.peg.append(peg_idx + self.peg[-1])

            return sample

        def create_sets_multi_classes(word_indexes, x_len, step):

            classes_x_samples = []
            for w_i in word_indexes:
                classes_x_samples.append(get_set_from_indexes(w_i, x_len, step))

            x_samples = []
            y_samples = []

            idx = 0
            progress_bar = tqdm(range(len(word_indexes)), ncols=800)
            progress_bar.set_description('Формирование массивов')
            for t in progress_bar:
                x_t = classes_x_samples[t]
                for i in range(len(x_t)):
                    x_samples.append(x_t[i])
                    y_samples.append(t)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total:
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)
            x_samples = np.array(x_samples)
            y_samples = np.array(y_samples)

            return x_samples, y_samples

        def get_sets(model, x, y):

            x_vector = []
            progress_bar = tqdm(x, ncols=800)
            progress_bar.set_description('Формирование массивов')
            idx = 0
            peg_idx = 0
            cls_idx = 0
            for i, text in enumerate(progress_bar):
                tmp = []
                for word in text:
                    tmp.append(model[word])
                x_vector.append(tmp)
                peg_idx += 1
                if cls_idx == round(sum(y[i]) / len(y[i]), 0):
                    self.peg.append(peg_idx-1)
                    cls_idx += 1
                if peg_idx == len(y):
                    self.peg.append(peg_idx - 1)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total:
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)

            return np.array(x_vector), np.array(y)

        if folder_name == None:
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]

        txt_list = []
        for _, dirnames, filename in sorted(os.walk(working_folder)):

            folders = sorted(dirnames)
            folders_num = len(dirnames) if len(dirnames) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except:
                    IndexError

                files = sorted(os.listdir(temp_path))
                progress_bar = tqdm(files, ncols=800)
                if folders_num == 1:
                    description = f'Сохранение текстов'
                else:
                    description = f'Сохранение текстов из папки {folders[i]}'
                progress_bar.set_description(description)
                idx = 0
                several_files = False
                for file in progress_bar:
                    if progress_bar.total > 1:
                        if not several_files:
                            txt_list.append(read_text(os.path.join(temp_path, file)))
                            several_files = True
                        else:
                            txt_list[-1] + read_text(os.path.join(temp_path, file))
                    else:
                        txt_list.append(read_text(os.path.join(temp_path, file)))
                    if self.django_flag:
                        idx += 1
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total and i+1 == folders_num:
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)

            break

        if pymorphy:
            for i in range(len(txt_list)):
                txt_list[i] = apply_pymorphy(txt_list[i])

        filters = '–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff'
        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i + 1}']['tag'] == 'text_segmentation':
                open_tags = self.user_parameters['out'][f'output_{i + 1}']['parameters']['open_tags']
                close_tags = self.user_parameters['out'][f'output_{i + 1}']['parameters']['close_tags']
                tags = f'{open_tags} {close_tags}'
                for ch in filters:
                    if ch in set(tags):
                        filters = filters.replace(ch, '')
                break
        tokenizer = Tokenizer(num_words=max_words_count, filters=filters,
                              lower=True, split=' ', char_level=False, oov_token='<UNK>')
        tokenizer.fit_on_texts(txt_list)
        text_seq = tokenizer.texts_to_sequences(txt_list)
        self.sequences = text_seq
        self.tokenizer[f'input_{self.iter}'] = tokenizer

        if embedding:
            self.peg = []
            X = []
            Y = []
            for i, txt in enumerate(txt_list):
                for word in txt.split(' '):
                    X.append(word)
                    Y.append(i)
            x = get_set_from_indexes(X, x_len, step)
            y = get_set_from_indexes(Y, x_len, step)
            self.word2vec[f'input_{self.iter}'] = word2vec.Word2Vec(x, size=embedding_size, window=10, min_count=1, workers=10, iter=10)
            X, Y = get_sets(self.word2vec[f'input_{self.iter}'], x, y)
        else:
            X, Y = create_sets_multi_classes(text_seq, x_len, step)

            if bag_of_words:
                X = np.array(tokenizer.sequences_to_matrix(X.tolist()))

        self.source_shape[f'input_{self.iter}'] = X.shape[1:]
        self.source_datatype += f' {self._set_datatype(shape=X.shape)}'
        self.x_Scaler[f'input_{self.iter}'] = None

        if 'classification' in self.task_type.values():
            self.y_Cls = Y.astype('int')
            del Y

        return X

    def dataframe(self, file_name=[''], separator='', encoding='utf-8', x_cols='', scaler=['No Scaler', 'StandardScaler', 'MinMaxScaler']) -> np.ndarray:

        self.classes_names[f'input_{self.iter}'] = x_cols.split(' ')
        if separator:
            X = pd.read_csv(os.path.join(self.file_folder, file_name), sep=separator, encoding=encoding)
        else:
            X = pd.read_csv(os.path.join(self.file_folder, file_name), encoding=encoding)
        self.df[f'input_{self.iter}'] = X

        X = X[x_cols.split(' ')].to_numpy()

        self.source_shape[f'input_{self.iter}'] = X.shape[1:]
        self.source_datatype += f' {self._set_datatype(shape=X.shape)}'

        if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
            shape_x = X.shape
            X = X.reshape(-1, 1)
            if scaler == 'MinMaxScaler':
                self.x_Scaler[f'input_{self.iter}'] = MinMaxScaler()
            elif scaler == 'StandardScaler':
                self.x_Scaler[f'input_{self.iter}'] = StandardScaler()
            self.x_Scaler[f'input_{self.iter}'].fit(X)
            X = self.x_Scaler[f'input_{self.iter}'].transform(X)
            X = X.reshape(shape_x)
        else:
            self.x_Scaler[f'input_{self.iter}'] = None

        #Если надо работать с временными рядами
        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i+1}']['tag'] == 'timeseries':
                length = self.user_parameters['out'][f'output_{i+1}']['parameters']['length']
                batch_size = self.user_parameters['out'][f'output_{i + 1}']['parameters']['batch_size']
                generator = TimeseriesGenerator(X, X, length=length, stride=1, batch_size=batch_size)
                X = []
                for i in range(len(generator)):
                    X.append(generator[i][0])
                X = np.array(X)
                self.tsgenerator[f'input_{self.iter}'] = generator
            break

        return X

    def regression(self, y_col='') -> np.ndarray:

        y_col = y_col.split(' ')
        self.classes_names[f'output_{self.iter}'] = y_col
        self.num_classes[f'output_{self.iter}'] = len(y_col)

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i+1}']['tag'] == 'dataframe':
                Y = self.df[f'input_{i+1}'][y_col].to_numpy()
                if self.user_parameters['inp'][f'input_{i + 1}']['parameters']['scaler'] in ['MinMaxScaler', 'StandardScaler']:
                    y_shape = Y.shape
                    Y = Y.reshape(-1, 1)
                    Y = self.x_Scaler[f'input_{i+1}'].transform(Y)
                    Y = Y.reshape(y_shape)

        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.peg = [0]
        for ratio in self.divide_ratio[1][:-1]:
            self.peg.append(self.peg[-1] + int(round(len(Y) * ratio, 0)))
        self.peg.append(len(Y))

        return Y

    def timeseries(self, length=1, batch_size=1) -> np.ndarray:

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i+1}']['tag'] == 'dataframe':
                columns = self.user_parameters['inp'][f'input_{i+1}']['parameters']['x_cols']

                self.classes_names[f'output_{self.iter}'] = columns.split(' ')
                self.num_classes[f'output_{self.iter}'] = len(columns.split(' '))

                Y = []
                for j in range(len(self.tsgenerator[f'input_{i+1}'])):
                    Y.append(self.tsgenerator[f'input_{i+1}'][j][1])
                Y = np.array(Y)
                if self.user_parameters['inp'][f'input_{i+1}']['parameters']['scaler'] in ['MinMaxScaler', 'StandardScaler']:
                    y_shape = Y.shape
                    Y = Y.reshape(-1, 1)
                    Y = self.x_Scaler[f'input_{i+1}'].transform(Y)
                    Y = Y.reshape(y_shape)

        self.y_Scaler[f'output_{self.iter}'] = None
        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.peg = [0]
        for ratio in self.divide_ratio[1][:-1]:
            self.peg.append(self.peg[-1] + int(round(len(Y) * ratio, 0)))
        self.peg.append(len(Y))

        return Y

    def audio(self, folder_name=[''], length=11025, step=2205,
              scaler=['No Scaler', 'StandardScaler', 'MinMaxScaler'], audio_signal=True, chroma_stft=False, mfcc=False, rms=False,
              spectral_centroid=False, spectral_bandwidth=False, spectral_rolloff=False, zero_crossing_rate=False) -> np.ndarray:

        def call_librosa(feature, section, sr):

            if feature in ['chroma_stft','mfcc','spectral_centroid','spectral_bandwidth','spectral_rolloff']:
                array = getattr(librosafeature, feature)(y=section, sr=sr)
            elif feature == 'rms':
                array = getattr(librosafeature, feature)(y=section)[0]
            elif feature == 'zero_crossing_rate':
                array = getattr(librosafeature, feature)(y=section)

            return array

        def wav_to_features(section):

            out = []
            if audio_signal:
                out.append(section)
            for feature in list_features:
                if feature == 'chroma_stft' or feature == 'mfcc':
                    arr = call_librosa(feature, section, sr)
                    for e in arr:
                        out.append(np.mean(e))
                else:
                    out.append(np.mean(call_librosa(feature, section, sr)))

            out = np.array(out)

            return out

        list_features = []
        features_str = ['chroma_stft', 'mfcc', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']
        features = [chroma_stft, mfcc, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
        for i, feature in enumerate(features):
            if feature is True:
                list_features.append(features_str[i])

        if folder_name == None:
            working_folder = self.file_folder
        else:
            working_folder = os.path.join(self.file_folder, folder_name)
        self.peg = [0]
        Y = np.array([])

        for _, dirnames, filename in sorted(os.walk(working_folder)):

            folders = sorted(dirnames)
            folders_num = len(dirnames) if len(dirnames) != 0 else 1
            for i in range(folders_num):
                temp_path = working_folder
                try:
                    temp_path = os.path.join(working_folder, folders[i])
                except:
                    IndexError

                files = [os.path.join(temp_path, wavfile) for wavfile in
                            sorted(os.listdir(temp_path))]

                if folders_num == 1:
                    description = f'Сохранение аудиофайлов'
                else:
                    description = f'Сохранение аудиофайлов из папки {folders[i]}'

                progress_bar = tqdm(files, ncols=800)
                progress_bar.set_description(description)
                out_vectors = []
                idx = 0
                peg_idx = 0
                for file in progress_bar:
                    y, sr = librosaload(file)
                    while (len(y) >= length):
                        section = y[:length]
                        section = np.array(section)
                        out = wav_to_features(section)
                        out_vectors.append(out)
                        y = y[step:]
                        peg_idx += 1
                    idx += 1
                    if self.django_flag:
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total and i+1 == folders_num:
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)
                out_vectors = np.array(out_vectors)
                try:
                    X = np.vstack((X, out_vectors))
                except NameError:
                    X = out_vectors
                Y = np.append(Y, np.full(out_vectors.shape[0], fill_value=(i)))
                self.peg.append(peg_idx + self.peg[-1])
            break

        self.source_shape[f'input_{self.iter}'] = X.shape[1:]
        self.source_datatype += f' {self._set_datatype(shape=X.shape)}'

        if scaler == 'MinMaxScaler' or scaler == 'StandardScaler':
            shape_x = X.shape
            X = X.reshape(-1, 1)
            if scaler == 'MinMaxScaler':
                self.x_Scaler = MinMaxScaler()
            elif scaler == 'StandardScaler':
                self.x_Scaler = StandardScaler()
            self.x_Scaler.fit(X)
            X = self.x_Scaler.transform(X)
            X = X.reshape(shape_x)
        else:
            self.x_Scaler[f'input_{self.iter}'] = None

        if 'classification' in self.task_type.values():
            self.y_Cls = Y.astype('int')
            del Y

        return X

    def classification(self, one_hot_encoding=[True, False]) -> np.ndarray:

        Y = self.y_Cls
        self.classes_names[f'output_{self.iter}'] = [folder for folder in sorted(os.listdir(self.file_folder))] # нет информации о выбранной пользователем папке. с другой стороны - надо ли..
        self.num_classes[f'output_{self.iter}'] = len(np.unique(Y, axis=0))
        self.y_Scaler[f'output_{self.iter}'] = None

        if one_hot_encoding:
            Y = utils.to_categorical(Y, len(np.unique(Y)))
            self.one_hot_encoding[f'output_{self.iter}'] = True
        else:
            self.one_hot_encoding[f'output_{self.iter}'] = False

        return Y

    def text_segmentation(self, open_tags='', close_tags='') -> np.ndarray:

        def get01XSamples(tok_agreem, tags_index):
            tags01 = []
            indexes = []

            for agreement in tok_agreem:
                tag_place = [0 for _ in range(len(open_tags.split(' ')))]
                for ex in agreement:
                    if ex in tags_index:
                        place = np.argwhere(tags_index == ex)
                        if len(place) != 0:
                            if place[0][0] < len(open_tags.split(' ')):
                                tag_place[place[0][0]] = 1
                            else:
                                tag_place[place[0][0] - len(open_tags.split(' '))] = 0
                    else:
                        tags01.append(
                            tag_place.copy())
                        indexes.append(ex)

            return indexes, tags01

        def get_set_from_indexes(word_indexes, x_len, step):

            sample = []
            words_len = len(word_indexes)

            index = 0
            while index + x_len <= words_len:
                sample.append(word_indexes[index:index + x_len])
                index += step

            return sample

        def get_sets(model, x, y):

            x_vector = []
            progress_bar = tqdm(x, ncols=800)
            progress_bar.set_description('Формирование массивов')
            idx = 0
            for text in progress_bar:
                tmp = []
                for word in text:
                    tmp.append(model[word])
                x_vector.append(tmp)
                if self.django_flag:
                    idx += 1
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    if idx == progress_bar.total:
                        self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                    else:
                        self.Exch.print_progress_bar(progress_bar_status)

            return np.array(x_vector), np.array(y)

        self.num_classes[f'output_{self.iter}'] = len(open_tags)
        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.y_Scaler[f'output_{self.iter}'] = None
        tags = open_tags.split(' ') + close_tags.split(' ')

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i+1}']['tag'] == 'text':
                x_len = self.user_parameters['inp'][f'input_{i+1}']['parameters']['x_len']
                step = self.user_parameters['inp'][f'input_{i+1}']['parameters']['step']
                tags_indexes = np.array([self.tokenizer[f'input_{i+1}'].word_index[k] for k in tags])
                break

        _, y_data = get01XSamples(self.sequences, tags_indexes)

        # X = get_set_from_indexes(x_data, x_len, step)
        Y = get_set_from_indexes(y_data, x_len, step)

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'text':
                if self.user_parameters['inp'][f'input_{i + 1}']['parameters']['embedding']:
                    reversed_tok = {}
                    for key, value in self.tokenizer.word_index.items():
                        reversed_tok[value] = key
                    text = []
                    for lst in self.sequences:
                        tmp = [reversed_tok.get(letter) for letter in lst]
                        text.append(tmp)
                    _, Y = get_sets(self.word2vec[f'input_{i+1}'], text, Y)
                    break
        Y = np.array(Y)

        return Y

    def segmentation(self, folder_name=[''], mask_range=10, input_type = ['Ручной ввод', 'Автоматический поиск', 'Файл аннотации'],
                     classes_dict={'название класса': [0, 0, 0]}) -> np.ndarray:

        def load_image(img_path, shape):

            img = load_img(img_path, target_size=shape)
            array = img_to_array(img)

            return array.astype('uint8')

        def cluster_to_ohe(image):

            image = image.reshape(-1, 3)
            km = KMeans(n_clusters=num_classes)
            km.fit(image)
            labels = km.labels_
            cl_cent = km.cluster_centers_.astype('uint8')[:max(labels) + 1]
            cl_mask = utils.to_categorical(labels, max(labels) + 1)
            cl_mask = cl_mask.reshape(shape[0], shape[1], cl_mask.shape[-1])

            mask_ohe = np.zeros(shape)
            for k, (name, rgb) in enumerate(classes_dict.items()):
                mask = np.zeros(shape)

                for j, cl_rgb in enumerate(cl_cent):
                    if rgb[0] in range(cl_rgb[0] - mask_range, cl_rgb[0] + mask_range) and \
                            rgb[1] in range(cl_rgb[1] - mask_range, cl_rgb[1] + mask_range) and \
                            rgb[2] in range(cl_rgb[2] - mask_range, cl_rgb[2] + mask_range):
                        mask = cl_mask[:, :, j]

                if k == 0:
                    mask_ohe = mask
                else:
                    mask_ohe = np.dstack((mask_ohe, mask))

            return mask_ohe

        self.classes_names[f'output_{self.iter}'] = list(classes_dict.keys())
        self.classes_colors[f'output_{self.iter}'] = list(classes_dict.values())
        num_classes = len(list(classes_dict.keys()))
        self.num_classes[f'output_{self.iter}'] = num_classes
        self.one_hot_encoding[f'output_{self.iter}'] = False
        self.y_Scaler[f'output_{self.iter}'] = None

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i+1}']['tag'] == 'images':
                height = self.user_parameters['inp'][f'input_{i+1}']['parameters']['height']
                width = self.user_parameters['inp'][f'input_{i+1}']['parameters']['width']
                shape = (height, width)
                break

        if folder_name == None:
            folder_name = self.file_folder
        else:
            folder_name = os.path.join(self.file_folder, folder_name)

        Y = []

        for _, dirnames, filename in sorted(os.walk(folder_name)):

            folders = sorted(dirnames)
            folders_num = len(dirnames) if len(dirnames) != 0 else 1
            for i in range(folders_num):
                temp_path = folder_name
                try:
                    temp_path = os.path.join(folder_name, folders[i])
                except:
                    IndexError

                files = sorted(os.listdir(temp_path))

                progress_bar = tqdm(files, ncols=800)
                progress_bar.set_description(f'Сохранение масок сегментации')
                idx = 0
                for file in progress_bar:
                    image = load_image(os.path.join(folder_name, file), shape)
                    image_ohe = cluster_to_ohe(image)
                    Y.append(image_ohe)
                    if self.django_flag:
                        idx += 1
                        progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                               f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                        if idx == progress_bar.total and i+1 == folders_num:
                            self.Exch.print_progress_bar(progress_bar_status, stop_flag=True)
                        else:
                            self.Exch.print_progress_bar(progress_bar_status)
            break
        Y = np.array(Y)

        return Y

    def object_detection(self) -> np.ndarray:

        def make_y(real_boxes, num_classes):

            anchors = np.array(
                [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
            num_layers = 3
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

            real_boxes = np.array(real_boxes, dtype='float32')
            input_shape = np.array((height, width), dtype='int32')

            boxes_wh = real_boxes[..., 2:4] * input_shape

            cells = [13, 26, 52]
            y_true = [np.zeros((cells[l], cells[l], len(anchor_mask[l]), 5 + num_classes), dtype='float32') for l in
                      range(num_layers)]
            box_area = boxes_wh[:, 0] * boxes_wh[:, 1]

            anchor_area = anchors[:, 0] * anchors[:, 1]
            for r in range(len(real_boxes)):
                correct_anchors = []
                for elem in anchors:
                    correct_anchors.append([min(elem[0], boxes_wh[r][0]), min(elem[1], boxes_wh[r][1])])
                correct_anchors = np.array(correct_anchors)
                correct_anchors_area = correct_anchors[:, 0] * correct_anchors[:, 1]
                iou = correct_anchors_area / (box_area[r] + anchor_area - correct_anchors_area)
                best_anchor = np.argmax(iou, axis=-1)

                for l in range(num_layers):
                    if best_anchor in anchor_mask[l]:
                        i = np.floor(real_boxes[r, 0] * cells[l]).astype('int32')
                        j = np.floor(real_boxes[r, 1] * cells[l]).astype('int32')
                        k = anchor_mask[l].index(best_anchor)
                        c = real_boxes[r, 4].astype('int32')
                        y_true[l][j, i, k, 0:4] = real_boxes[r, 0:4]
                        y_true[l][j, i, k, 4] = 1
                        y_true[l][j, i, k, 5 + c] = 1
                        break

            return y_true[0], y_true[1], y_true[2]

        for i in range(len(self.user_parameters['inp'])):
            if self.user_parameters['inp'][f'input_{i + 1}']['tag'] == 'images':
                height = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['height']
                width = self.user_parameters['inp'][f'input_{i + 1}']['parameters']['width']
                break

        data = {}
        class_names = {}

        # obj.data
        with open(os.path.join(self.file_folder, 'obj.data'), 'r') as dt:
            d = dt.read()
        for elem in d.split('\n'):
            if elem:
                elem = elem.split(' = ')
                data[elem[0]] = elem[1]
        num_classes = int(data['classes'])
        self.num_classes[f'output_{self.iter}'] = num_classes

        # obj.names
        with open(os.path.join(self.file_folder, data["names"].split("/")[-1]), 'r') as dt:
            names = dt.read()
        for i, elem in enumerate(names.split('\n')):
            if elem:
                class_names[i] = elem

        # list of txt
        txt_list = []
        with open(os.path.join(self.file_folder, data["train"].split("/")[-1]), 'r') as dt:
            imgs = dt.read()
        for elem in imgs.split('\n'):
            if elem:
                idx = elem.rfind('.')
                elem = elem.replace(elem[idx:], '.txt')
                txt_list.append(os.path.join(*elem.split('/')[1:]))

        input_1 = []
        input_2 = []
        input_3 = []

        # read txt files + make y's
        for txt_file in sorted(txt_list):
            with open(os.path.join(self.file_folder, txt_file), 'r') as txt:
                bb_file = txt.read()
            real_bb = []
            for elem in bb_file.split('\n'):
                tmp = []
                if elem:
                    for num in elem.split(' '):
                        tmp.append(float(num))
                    real_bb.append(tmp)
            real_bb = np.array(real_bb)
            real_bb = real_bb[:, [1, 2, 3, 4, 0]]
            out1, out2, out3 = make_y(real_boxes=real_bb, num_classes=num_classes)
            input_1.append(out1)
            input_2.append(out2)
            input_3.append(out3)

        input_1 = np.array(input_1)
        input_2 = np.array(input_2)
        input_3 = np.array(input_3)

        return input_1, input_2, input_3

    def prepare_user_dataset(self, dataset_dict: dict, is_save=True):

        cur_time = time()
        if self.django_flag:
            for key, value in dataset_dict["inputs"].items():

                for param_key, param_value in value["parameters"].items():
                    try:
                        if (param_key == "folder_name"):
                            continue
                        if (param_value == 'true' or param_value == 'on'):
                            dataset_dict["inputs"][key]["parameters"][param_key] = True
                        elif (param_value == 'false'):
                            dataset_dict["inputs"][key]["parameters"][param_key] = False
                        elif param_value.isdigit():
                            dataset_dict["inputs"][key]["parameters"][param_key] = int(param_value)
                    except ValueError:
                        continue

            for key, value in dataset_dict["outputs"].items():

                for param_key, param_value in value["parameters"].items():
                    try:
                        if (param_key == "folder_name"):
                            continue
                        if (param_value == 'true' or param_value == 'on'):
                            dataset_dict["outputs"][key]["parameters"][param_key] = True
                        elif (param_value == 'false'):
                            dataset_dict["outputs"][key]["parameters"][param_key] = False
                        else:
                            dataset_dict["outputs"][key]["parameters"][param_key] = int(param_value)
                    except ValueError:
                        continue

            for key, value in dataset_dict["parameters"].items():

                try:
                    if (param_key == "folder_name"):
                        continue
                    if (value == 'true' or value == 'on'):
                        dataset_dict["parameters"][key] = True
                    elif (value == 'false'):
                        dataset_dict["parameters"][key] = False
                    else:
                        dataset_dict["parameters"][key] = int(value)
                except ValueError:
                    continue

        self.name = dataset_dict['parameters']['name']
        self.user_tags = dataset_dict['parameters']['user_tags']
        self.divide_ratio[1] = (dataset_dict['parameters']['train_part'], dataset_dict['parameters']['val_part'], dataset_dict['parameters']['test_part'])

        self.user_parameters['inp'] = dataset_dict['inputs']
        self.user_parameters['out'] = dataset_dict['outputs']

        for i in range(len(self.user_parameters['inp'])):
            self.tags[f'input_{i + 1}'] = dataset_dict['inputs'][f'input_{i + 1}']['tag']
        for i in range(len(self.user_parameters['out'])):
            self.tags[f'output_{i + 1}'] = dataset_dict['outputs'][f'output_{i + 1}']['tag']
            self.task_type[f'output_{i + 1}'] = dataset_dict['outputs'][f'output_{i + 1}']['task_type']

        for i in range(len(self.user_parameters['inp'])):
            self.iter = i + 1
            self.X[f'input_{i+1}'] = {'data_name': self.user_parameters['inp'][f'input_{i+1}']['name'], 'data': getattr(self, self.user_parameters['inp'][f'input_{i+1}']['tag'])(**self.user_parameters['inp'][f'input_{i+1}']['parameters'])}

        for i in range(len(self.user_parameters['out'])):
            self.iter = i + 1
            if self.user_parameters['out'][f'output_{i+1}']['tag'] == 'object_detection':
                outputs = getattr(self, self.user_parameters['out'][f'output_{i+1}']['tag'])(**self.user_parameters['out'][f'output_{i+1}']['parameters'])
                for k in range(3):
                    self.Y[f'output_{i+k+1}'] = {'data_name': self.user_parameters['out'][f'output_{i+1}']['name'], 'data': outputs[k]}
                    self.tags[f'output_{i+k+1}'] = dataset_dict['outputs'][f'output_{i+1}']['tag']
                    self.task_type[f'output_{i+k+1}'] = dataset_dict['outputs'][f'output_{i+1}']['task_type']
            else:
                self.Y[f'output_{i+1}'] = {'data_name': self.user_parameters['out'][f'output_{i+1}']['name'], 'data': getattr(self, self.user_parameters['out'][f'output_{i+1}']['tag'])(**self.user_parameters['out'][f'output_{i+1}']['parameters'])}

        # Train/Val/Test split
        train_mask = []
        val_mask = []
        test_mask = []

        for i in range(len(self.peg) - 1):
            indices = np.arange(self.peg[i], self.peg[i + 1]).tolist()
            train_len = int(self.divide_ratio[1][0] * len(indices))
            val_len = int(self.divide_ratio[1][1] * len(indices))
            train_mask.extend(indices[:train_len])
            val_mask.extend(indices[train_len:train_len + val_len])
            test_mask.extend(indices[train_len + val_len:])

        for i in range(len(self.user_parameters['out'])):
            if self.user_parameters['out'][f'output_{i+1}']['tag'] == 'timeseries':
                length = self.user_parameters['out'][f'output_{i+1}']['parameters']['length']
                train_mask = train_mask[:-length]
                val_mask = val_mask[:-length]

        if not dataset_dict['parameters']['preserve_sequence']:
            random.shuffle(train_mask)
            random.shuffle(val_mask)
            random.shuffle(test_mask)

        for inp in self.X.keys():
            self.X[inp]['data'] = (self.X[inp]['data'][train_mask], self.X[inp]['data'][val_mask],
                                   self.X[inp]['data'][test_mask])
        for out in self.Y.keys():
            self.Y[out]['data'] = (self.Y[out]['data'][train_mask], self.Y[out]['data'][val_mask],
                                   self.Y[out]['data'][test_mask])

        inp_datatype = []
        for inp in self.X.keys():
            self.input_shape[inp] = self.X[inp]['data'][0].shape[1:]
            inp_datatype.append(self._set_datatype(shape=self.X[inp]['data'][0].shape))

        for out in self.Y.keys():
            self.output_shape[out] = self.Y[out]['data'][0].shape[1:]
            self.output_datatype[out] = self._set_datatype(shape=self.Y[out]['data'][0].shape)

        self.input_datatype = ' '.join(inp_datatype)

        if not self.django_flag:
            print(f'Формирование массивов завершено. Времени затрачено: {round(time() - cur_time, 2)} сек.')
            x = ['x_train', 'x_val', 'x_test']
            y = ['y_train', 'y_val', 'y_test']
            for inp in self.X.keys():
                for i, item in enumerate(self.X[inp]['data']):
                    if isinstance(item, np.ndarray):
                        print(f'Размерность {inp} - {x[i]}: {self.X[inp]["data"][i].shape}')
            for out in self.Y.keys():
                for i, item in enumerate(self.Y[out]['data']):
                    if isinstance(item, np.ndarray):
                        print(f'Размерность {out} - {y[i]}: {self.Y[out]["data"][i].shape}')

        temp_attributes = ['iter', 'model_gensim'] # 'y_Cls' 'sequences' 'peg', 'df'
        for item in temp_attributes:
            if hasattr(self, item):
                delattr(self, item)

        self.dts_prepared = True
        if is_save:
            print('Идёт сохранение датасета.')
            directory = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'TerraAI', 'datasets')
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(f"{directory}/{self.name}.trds", "wb") as f:
                dill.dump(self, f)
            tzinfo = timezone('Europe/Moscow')
            now = datetime.now().astimezone(tzinfo)
            dt_string = now.isoformat()
            data = {}
            data['name'] = self.name
            data['source'] = self.source
            data['tags'] = list(self.tags.values())
            data['date'] = dt_string
            data['size'] = self._get_size(f'{directory}/{self.name}.trds')
            with open(f'{directory}/{self.name}.trds.json', 'w') as fp:
                json.dump(data, fp)
            print(f'Датасет сохранен в файл {directory}/{self.name}.trds')
            print(f'Json сохранен в файл {directory}/{self.name}.trds.json')

        return self

    # def print_data(self, type):
    #     if type == 'images':
    #         fig, ax = plt.subplots(1, self.num_classes, figsize=(self.num_classes * 3, 6))
    #         for i in range(self.num_classes):
    #             arr = np.zeros(3, dtype='int8')
    #             arr[i] = 1
    #             index = np.where(self.y_Train == arr)[0]
    #             index = np.random.choice(index, 1)[0]
    #             ax[i].imshow(self.x_Train[index])
    #             ax[i].set_title(f'{i}: {self.classes_names[i]}')
    #
    #     return self