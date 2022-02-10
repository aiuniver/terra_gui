import os
import numpy as np
import joblib
import json
import pandas as pd
import tensorflow as tf
import shutil
import zipfile
import tempfile
import re
import h5py

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from PIL import Image
from pathlib import Path
from datetime import datetime
from IPython.display import display

# from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.utils import get_tempdir
from terra_ai.datasets.preprocessing import CreatePreprocessing
# from terra_ai.datasets import arrays_create
# from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.data.datasets.dataset import DatasetData, DatasetCommonPathsData, VersionData, DatasetVersionPathsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice
from terra_ai.data.presets.datasets import KerasInstructions
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, DATASET_VERSION_EXT, VERSION_CONFIG
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
    version_data = None
    preprocessing = CreatePreprocessing()

    def __init__(self, alias: str = ''):

        if alias.endswith('.' + DATASET_EXT):
            self.dataset_paths_data = DatasetCommonPathsData(basepath=get_tempdir())
            self.parent_dataset_paths_data = DatasetCommonPathsData(
                basepath=TERRA_PATH.joinpath(alias)
            )
            shutil.copy(self.parent_dataset_paths_data.basepath.joinpath(DATASET_CONFIG),
                        self.dataset_paths_data.basepath.joinpath(DATASET_CONFIG))
            with open(self.dataset_paths_data.basepath.joinpath(DATASET_CONFIG), 'r') as cfg:
                config = json.load(cfg)
            self.dataset_data = DatasetData(**config)
            self.version_paths_data = None
        elif alias.endswith('.trds_terra'):
            self.dataset_paths_data = DatasetCommonPathsData(basepath=get_tempdir())
            pass
        elif alias.endswith('.keras'):
            for d_config in DatasetsGroups[0]['datasets']:
                if d_config['alias'] == re.search(r"[A-Za-z0-9_-]+", alias)[0]:
                    self.dataset_data = DatasetData(**d_config)
                    break

        # self.dataset_data: DatasetData = data
        # if self.dataset_data.group != DatasetGroupChoice.keras:
        #     self.version_paths_data = None
        #     self.dataset_paths_data = DatasetCommonPathsData(basepath=datasets_path)
        #     self.parent_dataset_paths_data = DatasetCommonPathsData(basepath=TERRA_PATH.joinpath('datasets'))

        # else:
        #     self.preprocessing = CreatePreprocessing()

    def __str__(self):

        dataset = f'{self.dataset_data.alias} / {self.dataset_data.name}'\
            if self.__dict__.get('dataset_data') else "не выбран"
        version = f'{self.version_data.alias} / {self.version_data.name}'\
            if self.__dict__.get('version_data') else "не выбрана"

        # dictio = {'alias': [self.dataset_data.alias if self.__dict__.get("dataset_data") else "не выбран",
        #                     self.version_data.alias if self.__dict__.get("version_data") else "не выбран"],
        #           'Название': [self.dataset_data.name if self.__dict__.get("dataset_data") else "не выбран",
        #                        self.version_data.name if self.__dict__.get("version_data") else "не выбран"]}
        #
        # return pd.DataFrame(dictio, index=['Датасет', 'Версия'])

        return f'Датасет: {dataset}.\n' \
               f'Версия: {version}.'

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
                with open(d_path.joinpath(VERSION_CONFIG), 'r') as config:
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

    def generator_common(self, split_name):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe[split_name])):
            for inp_id in self.version_data.inputs.keys():
                tmp = []
                for col_name, data in self.instructions[inp_id].items():
                    if data['put_type'] in PATH_TYPE_LIST:
                        sample = self.version_paths_data.sources.joinpath(self.dataframe[split_name].loc[idx, col_name])
                    else:
                        sample = self.dataframe[split_name].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[inp_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                inputs[str(inp_id)] = np.concatenate(tmp, axis=0)

            for out_id in self.version_data.outputs.keys():
                tmp = []
                for col_name, data in self.instructions[out_id].items():
                    if data['put_type'] in PATH_TYPE_LIST:
                        sample = self.version_paths_data.basepath.joinpath(
                            self.dataframe[split_name].loc[idx, col_name]
                        )
                    else:
                        sample = self.dataframe[split_name].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[out_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                outputs[str(out_id)] = np.concatenate(tmp, axis=0)

            yield inputs, outputs
    def generator(self, inputs, outputs, service=None):

        for i in range(len(inputs)):

            with h5py.File(self.paths.arrays.joinpath('dataset.h5'), 'r') as hdf:
                # full_inp_arrays = [hdf[path][()] for path in inputs[i]]
                # full_out_arrays = [hdf[path][()] for path in outputs[i]]

                # inp_dict = {str(j + 1): full_inp_arrays[j] for j in range(len(full_inp_arrays))}
                # out_dict = {str(len(inputs[i]) + j + 1): full_out_arrays[j] for j in range(len(full_out_arrays))}

                inp_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in inputs[i]}
                out_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in outputs[i]}

                if self.data.service:

                    # full_srv_arrays = [hdf[path][()] for path in service[i]]
                    # srv_dict = {str(len(inputs[i]) + j + 1): full_srv_arrays[j] for j in range(len(service[i]))}

                    srv_dict = {elem.split('/')[1].split('_')[1]: hdf[elem][()] for elem in service[i]}
                    yield inp_dict, out_dict, srv_dict
                else:
                    yield inp_dict, out_dict

    def keras_datasets(self):

        (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()

        if self.data.alias in ['mnist', 'fashion_mnist']:
            x_train = x_train[..., None]
            x_val = x_val[..., None]
        for out in self.data.outputs.keys():
            if self.data.outputs[out].task == LayerOutputTypeChoice.Classification:
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))

        # for split in ['train', 'val']:
        #     for key in self.data.inputs.keys():
        #         self.X[split][str(key)] = globals()[f'x_{split}']
        #     for key in self.data.outputs.keys():
        #         self.Y[split][str(key)] = globals()[f'y_{split}']
        for key in self.data.inputs.keys():
            self.X['train'][str(key)] = x_train
            self.X['val'][str(key)] = x_val
        for key in self.data.outputs.keys():
            self.Y['train'][str(key)] = y_train
            self.Y['val'][str(key)] = y_val

    def prepare_dataset(self):

        if self.data.group == DatasetGroupChoice.keras:

            self.keras_datasets()

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
                if self.service[split]:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split], self.Y[split],
                                                                      self.service[split]))
                else:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split], self.Y[split]))

        elif self.data.group in [DatasetGroupChoice.terra, DatasetGroupChoice.custom]:

            for split in ['train', 'val']:
                self.dataframe[split] = pd.read_csv(os.path.join(self.paths.instructions, 'tables', f'{split}.csv'),
                                                    index_col=0)

            self.preprocessing.load_preprocesses(self.data.columns)

            if self.data.use_generator:

                # num_inputs = len(self.data.inputs)
                # num_outputs = len(self.data.outputs)

                # out_signature = [
                #     {str(x): tf.TensorSpec(shape=self.data.inputs[x].shape, dtype=self.data.inputs[x].dtype)
                #      for x in range(1, num_inputs + 1)},
                #     {str(x): tf.TensorSpec(shape=self.data.outputs[x].shape, dtype=self.data.outputs[x].dtype)
                #      for x in range(num_inputs + 1, num_inputs + num_outputs + 1)},
                #     ]
                # if self.data.service:
                #     out_signature.append(
                #         {str(x): tf.TensorSpec(shape=self.data.service[x].shape, dtype=self.data.service[x].dtype)
                #          for x in range(num_inputs + 1, num_inputs + num_outputs + 1)}
                #     )

                # for split_g in ['train', 'val']:
                #
                #     globals()[f'{split_g}_files_x'] = []
                #     globals()[f'{split_g}_files_y'] = []
                #     globals()[f'{split_g}_files_s'] = []
                #
                #     for idx in range(len(self.dataframe[split_g])):
                #         globals()[f'{split_g}_files_x'].append([f"{split_g}/id_{key}/{idx}" for key in self.data.inputs])
                #         globals()[f'{split_g}_files_y'].append([f"{split_g}/id_{key}/{idx}" for key in self.data.outputs])
                #         globals()[f'{split_g}_files_s'].append([f"{split_g}/id_{key}_service/{idx}" for key in self.data.service
                #                                                 if self.data.service])
                #
                #     globals()[f"{split_g}_parameters"] = {'inputs': globals()[f'{split_g}_files_x'],
                #                                           'outputs': globals()[f'{split_g}_files_y']}
                #     if self.data.service:
                #         globals()[f"{split_g}_parameters"].update([('service', globals()[f'{split_g}_files_s'])])

                with h5py.File(self.paths.arrays.joinpath('dataset.h5'), 'r') as hdf:

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

            else:

                for split in ['train', 'val']:
                    for index in self.data.inputs.keys():
                        self.X[split][str(index)] = joblib.load(os.path.join(self.paths.arrays, split, f'{index}.gz'))
                    for index in self.data.outputs.keys():
                        self.Y[split][str(index)] = joblib.load(os.path.join(self.paths.arrays, split, f'{index}.gz'))
                    for index in self.data.service.keys():
                        if self.data.service[index]:
                            self.service[split][str(index)] = joblib.load(os.path.join(self.paths.arrays,
                                                                                       split, f'{index}_service.gz'))

                for split in ['train', 'val']:
                    if self.service[split]:
                        self.dataset[split] = Dataset.from_tensor_slices((self.X[split], self.Y[split],
                                                                          self.service[split]))
                    else:
                        self.dataset[split] = Dataset.from_tensor_slices((self.X[split], self.Y[split]))

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

        with open(os.path.join(folder_path, VERSION_CONFIG), 'w') as cfg:
            json.dump(self.version_data.native(), cfg)
