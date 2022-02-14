import os
import numpy as np
import joblib
import json
import pandas as pd
import tensorflow as tf
import h5py

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from pathlib import Path
from datetime import datetime
from IPython.display import display

from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.dataset import DatasetData, DatasetVersionPathsData
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.presets.datasets import KerasInstructions
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, DATASET_VERSION_EXT, DATASET_VERSION_CONFIG
from terra_ai.data.presets.datasets import DatasetsGroups, VersionsGroups

TERRA_PATH = Path('G:\\Мой диск\\TerraAI\\datasets')


class PrepareDataset(object):
    """
    Класс для загрузки датасета.
    """

    dataframe: dict = {'train': None, 'val': None}
    instructions: dict = {}
    X: dict = {'train': {}, 'val': {}}
    Y: dict = {'train': {}, 'val': {}}
    service: dict = {'train': {}, 'val': {}}
    dataset: dict = {'train': None, 'val': None}
    preprocessing = CreatePreprocessing()

    def __init__(self, data: DatasetData, datasets_path=None):

        self.data = data
        self.dts_prepared = False
        if self.data.group != DatasetGroupChoice.keras:
            self.dataset_version_paths_data = DatasetVersionPathsData(basepath=datasets_path)
            self.preprocessing = CreatePreprocessing(dataset_path=datasets_path)

    def __str__(self):

        dataset = f'{self.data.alias} / {self.data.name}'\
            if self.__dict__.get('data') else "не выбран"
        # version = f'{self.version_data.alias} / {self.version_data.name}'\
        #     if self.__dict__.get('version_data') else "не выбрана"

        # dictio = {'alias': [self.dataset_data.alias if self.__dict__.get("dataset_data") else "не выбран",
        #                     self.version_data.alias if self.__dict__.get("version_data") else "не выбран"],
        #           'Название': [self.dataset_data.name if self.__dict__.get("dataset_data") else "не выбран",
        #                        self.version_data.name if self.__dict__.get("version_data") else "не выбран"]}
        #
        # return pd.DataFrame(dictio, index=['Датасет', 'Версия'])

        return f'Датасет: {dataset}.'# \
               # f'Версия: {version}.'

    @staticmethod
    def list_datasets(category='custom'):

        """
        Метод просмотра доступных к выбору датасетов.
        Input:
            category: str = "custom"
            Одна из трёх категорий датасетов: "custom", "terra", "keras".
        Пример:
        PrepareDataset().list_datasets("terra")
        """

        if category == 'custom':
            build_table = {'alias': [], 'Название': [], 'Задача': [], 'Дата создания': []}
            for d_path in Path(TERRA_PATH).glob('*.' + DATASET_EXT):  # В БУДУЩЕМ СДЕЛАТЬ TERRA_PATH.datasets
                with open(d_path.joinpath(DATASET_CONFIG), 'r') as config:
                    d_config = json.load(config)
                build_table['alias'].append('.'.join([d_config.get('alias'), DATASET_EXT])
                                            if d_config.get('alias') else '')
                build_table['Название'].append(d_config.get('name', ''))
                build_table['Задача'].append(d_config.get('architecture', ''))
                build_table['Дата создания'].append(
                    datetime.fromisoformat(d_config['date']).strftime('%d %b %Y, %H:%M:%S') if d_config.get('date',
                                                                                                            '') else '')
            dataframe = pd.DataFrame.from_dict(build_table)
            display(dataframe)
        elif category == 'keras':
            build_table = {'alias': [], 'Название': [], 'Задача': []}
            for d_config in DatasetsGroups[0]['datasets']:
                build_table['alias'].append('.'.join([d_config.get('alias'), 'keras']) if d_config.get('alias') else '')
                build_table['Название'].append(d_config.get('name', ''))
                build_table['Задача'].append(d_config.get('architecture', ''))
            dataframe = pd.DataFrame.from_dict(build_table)
            display(dataframe)
        elif category == 'terra':
            pass

        # print(dataframe)
        # return dataframe
        # display(dataframe) if hasattr(__builtins__, '__IPYTHON__') else print(dataframe.to_markdown())

    # @staticmethod
    # def list_keras_datasets():
    #
    #     build_table = {'alias': [], 'Название': [], 'Задача': []}
    #     for d_config in DatasetsGroups[0]['datasets']:
    #         build_table['alias'].append('.'.join([d_config.get('alias'), 'keras']) if d_config.get('alias') else '')
    #         build_table['Название'].append(d_config.get('name', ''))
    #         build_table['Задача'].append(d_config.get('architecture', ''))
    #     dataframe = pd.DataFrame.from_dict(build_table)
    #     display(dataframe)

    # @staticmethod
    # def list_preinstalled_datasets():
    #     pass

    def list_versions(self, alias: str = ''):

        """
        Метод просмотра доступных к выбору версий датасета.
        Inputs:
            alias(Optional): str - alias датасета.
        Допускается не указывать alias при вызове данного метода в случае его вызова от созданного экземпляра
        класса PrepareDataset() с выбранным датасетом. Пример:
        dataset = PrepareDataset('cars_dataset.trds')
        dataset.list_versions()
        """

        if not alias and not self.dataset_data:
            raise ValueError('Укажите alias датасета или выберите датасет.')
        elif not alias and self.dataset_data:
            alias = self.dataset_data.alias
            # alias = '.'.join([self.dataset_data.alias, self.dataset_data.group])

        build_table = {'alias': [], 'Название': [], 'Входы': [], 'Выходы': []}
        if self.dataset_data.group == 'trds':
            build_table.update({'Размер': [], 'Генератор': [], 'Дата создания': []})
            for d_path in Path(TERRA_PATH).joinpath('.'.join([alias, DATASET_EXT]), 'versions').glob('*.' + DATASET_VERSION_EXT):
                with open(d_path.joinpath(DATASET_VERSION_CONFIG), 'r') as config:
                    d_config = json.load(config)
                build_table['alias'].append(d_config.get('alias') if d_config.get('alias') else '')
                build_table['Название'].append(d_config.get('name', ''))
                build_table['Генератор'].append(d_config.get('use_generator', ''))
                build_table['Размер'].append(
                    str(round(d_config['size']['short'], 2)) + d_config['size']['unit'] if d_config.get('size', '') else '')
                build_table['Дата создания'].append(
                    datetime.fromisoformat(d_config['date']).strftime('%d %b %Y, %H:%M:%S') if d_config.get('date',
                                                                                                            '') else '')
                for put in [['inputs', 'Входы'], ['outputs', 'Выходы']]:
                    for idx, elem in enumerate(d_config[put[0]].values()):
                        build_table[put[1]].append(f"{put[1][:-1]} {idx + 1}: {elem['shape']}")
                max_len = max(len(build_table['Входы']), len(build_table['Выходы']))
                for put in build_table:
                    while not len(build_table[put]) == max_len:
                        build_table[put].append('')
        elif self.dataset_data.group == 'keras':
            for ver in VersionsGroups[0]['datasets'][0][alias]:
                build_table['alias'].append(ver.get('alias') if ver.get('alias') else '')
                build_table['Название'].append(ver['name'])
                for put in [['inputs', 'Входы'], ['outputs', 'Выходы']]:
                    for idx, elem in enumerate(ver[put[0]].values()):
                        build_table[put[1]].append(f"{put[1][:-1]} {idx + 1}: {elem['shape']}")
        elif self.dataset_data.group == 'terra':
            pass

        dataframe = pd.DataFrame().from_dict(build_table)
        display(dataframe)
        # display(dataframe) if hasattr(__builtins__, '__IPYTHON__') else print(dataframe.to_markdown(index=False))

    def generator(self, inputs, outputs, service=None):

        for i in range(len(inputs)):

            with h5py.File(self.dataset_version_paths_data.arrays.joinpath('dataset.h5'), 'r') as hdf:
                inp_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in inputs[i]}
                out_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in outputs[i]}

                if self.data.service:
                    srv_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in service[i]}
                    yield inp_dict, out_dict, srv_dict
                else:
                    yield inp_dict, out_dict

    def keras_datasets(self):

        (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()

        if self.data.alias in ['mnist', 'fashion_mnist'] and self.data.version.alias == 'add_dimension':
            x_train = x_train[..., None]
            x_val = x_val[..., None]
        elif self.data.version.alias == 'flatten':
            x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
            x_val = x_val.reshape((x_val.shape[0], np.prod(x_val.shape[1:])))
        y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)), dtype=np.uint8)
        y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)), dtype=np.uint8)

        x, y = {'train': {}, 'val': {}}, {'train': {}, 'val': {}}
        for key in self.data.inputs.keys():
            x['train'][str(key)] = x_train
            x['val'][str(key)] = x_val
        for key in self.data.outputs.keys():
            y['train'][str(key)] = y_train
            y['val'][str(key)] = y_val

        return x, y

    def prepare_dataset(self):

        if self.data.group == DatasetGroupChoice.keras:

            self.X, self.Y = self.keras_datasets()

            for put_id, data in KerasInstructions[self.data.alias].items():
                self.instructions[put_id] = data

            self.preprocessing.create_scaler(**{'put': 1, 'scaler': 'min_max_scaler',
                                                'min_scaler': 0, 'max_scaler': 1,
                                                'cols_names': f'1_{self.data.alias}'})
            self.preprocessing.preprocessing[1][f'1_{self.data.alias}'].fit(self.X['train']['1'].reshape(-1, 1))
            for key in self.X.keys():
                for inp in self.X[key]:
                    self.X[key][inp] = self.preprocessing.preprocessing[1][f'1_{self.data.alias}']\
                        .transform(self.X[key][inp].reshape(-1, 1)).reshape(self.X[key][inp].shape)

            for split in ['train', 'val']:
                self.dataset[split] = Dataset.from_tensor_slices((self.X[split], self.Y[split]))

        elif self.data.group in [DatasetGroupChoice.terra, DatasetGroupChoice.custom]:

            for split in ['train', 'val']:
                self.dataframe[split] = pd.read_csv(
                    self.dataset_version_paths_data.instructions.joinpath('tables', f'{split}.csv'),
                    index_col=0
                )

            self.preprocessing.load_preprocesses(self.data.columns)

            with h5py.File(self.dataset_version_paths_data.arrays.joinpath('dataset.h5'), 'r') as hdf:
                out_signature = [{}, {}]

                for key in self.data.inputs:
                    if f"train/id_{key}/1" in hdf:
                        out_signature[0].update({str(key): tf.TensorSpec(shape=self.data.inputs[key].shape,
                                                                         dtype=self.data.inputs[key].dtype)})
                for key in self.data.outputs:
                    if f"train/id_{key}/0" in hdf:
                        out_signature[1].update({str(key): tf.TensorSpec(shape=self.data.outputs[key].shape,
                                                                         dtype=self.data.outputs[key].dtype)})

                if self.data.service:
                    out_signature.append({})
                    for key in self.data.service:
                        if f"train/id_{key}_service/0" in hdf:
                            out_signature[2].update({str(key): tf.TensorSpec(shape=self.data.service[key].shape,
                                                                             dtype=self.data.service[key].dtype)})

                for split_g in ['train', 'val']:
                    globals()[f'{split_g}_files_x'] = []
                    globals()[f'{split_g}_files_y'] = []
                    globals()[f'{split_g}_files_s'] = []

                    for idx in range(len(self.dataframe[split_g])):
                        globals()[f'{split_g}_files_x'].append(
                            [f"{split_g}/id_{key}/{idx}" for key in self.data.inputs if
                             f"{split_g}/id_{key}/{idx}" in hdf])
                        globals()[f'{split_g}_files_y'].append(
                            [f"{split_g}/id_{key}/{idx}" for key in self.data.outputs if
                             f"{split_g}/id_{key}/{idx}" in hdf])
                        globals()[f'{split_g}_files_s'].append(
                            [f"{split_g}/id_{key}_service/{idx}" for key in self.data.service
                             if self.data.service and f"{split_g}/id_{key}/{idx}" in hdf])

                        globals()[f"{split_g}_parameters"] = {'inputs': globals()[f'{split_g}_files_x'],
                                                              'outputs': globals()[f'{split_g}_files_y']}
                        if self.data.service:
                            globals()[f"{split_g}_parameters"].update([('service', globals()[f'{split_g}_files_s'])])\

                    globals()[f"{split_g}_parameters"] = {'inputs': globals()[f'{split_g}_files_x'],
                                                          'outputs': globals()[f'{split_g}_files_y']}
                    if self.data.service:
                        globals()[f"{split_g}_parameters"].update([('service', globals()[f'{split_g}_files_s'])])

            self.dataset['train'] = Dataset.from_generator(lambda: self.generator(**globals()[f"train_parameters"]),
                                                           output_signature=tuple(out_signature))

            self.dataset['val'] = Dataset.from_generator(lambda: self.generator(**globals()[f"val_parameters"]),
                                                         output_signature=tuple(out_signature))

        self.dts_prepared = True

    def deploy_export(self, folder_path: str):

        parameters_path = os.path.join(folder_path, 'instructions', 'parameters')
        os.makedirs(parameters_path, exist_ok=True)
        for put in self.instructions.keys():
            for col_name, data in self.instructions[put].items():
                with open(os.path.join(parameters_path, f'{col_name}.json'), 'w') as instr:
                    json.dump(data, instr)

        preprocessing_path = os.path.join(folder_path, 'preprocessing')
        os.makedirs(preprocessing_path, exist_ok=True)
        for put, proc in self.preprocessing.preprocessing.items():
            for col_name, obj in proc.items():
                if obj:
                    folder_dir = os.path.join(preprocessing_path, str(put))
                    os.makedirs(folder_dir, exist_ok=True)
                    joblib.dump(obj, os.path.join(folder_dir, f'{col_name}.gz'))

        with open(os.path.join(folder_path, DATASET_VERSION_CONFIG), 'w') as cfg:
            json.dump(self.data.native(), cfg)
