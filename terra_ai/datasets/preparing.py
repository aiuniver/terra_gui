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

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from PIL import Image
from pathlib import Path
from datetime import datetime
from IPython.display import display

from terra_ai.utils import decamelize
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.data.datasets.dataset import DatasetData, DatasetPathsData, VersionData, VersionPathsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice
from terra_ai.data.presets.datasets import KerasInstructions
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, VERSION_EXT, VERSION_CONFIG
from terra_ai.data.presets.datasets import DatasetsGroups, VersionsGroup

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
            self.dataset_paths_data = DatasetPathsData(basepath=Path(tempfile.mkdtemp()))
            self.parent_dataset_paths_data = DatasetPathsData(
                basepath=TERRA_PATH.joinpath(alias)
            )
            shutil.copy(self.parent_dataset_paths_data.basepath.joinpath(DATASET_CONFIG),
                        self.dataset_paths_data.basepath.joinpath(DATASET_CONFIG))
            with open(self.dataset_paths_data.basepath.joinpath(DATASET_CONFIG), 'r') as cfg:
                config = json.load(cfg)
            self.dataset_data = DatasetData(**config)
            self.version_paths_data = None
        elif alias.endswith('.trds_terra'):
            self.dataset_paths_data = DatasetPathsData(basepath=Path(tempfile.mkdtemp()))
            pass
        elif alias.endswith('.keras'):
            for d_config in DatasetsGroups[0]['datasets']:
                if d_config['alias'] == re.search(r"[A-Za-z0-9_-]+", alias)[0]:
                    self.dataset_data = DatasetData(**d_config)
                    break

        # self.dataset_data: DatasetData = data
        # if self.dataset_data.group != DatasetGroupChoice.keras:
        #     self.version_paths_data = None
        #     self.dataset_paths_data = DatasetPathsData(basepath=datasets_path)
        #     self.parent_dataset_paths_data = DatasetPathsData(basepath=TERRA_PATH.joinpath('datasets'))

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
            for d_path in Path(TERRA_PATH).joinpath('.'.join([alias, DATASET_EXT]), 'versions').glob('*.' + VERSION_EXT):
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
            for ver in VersionsGroup[0]['datasets'][0][alias]:
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

    def generator_object_detection(self, split_name):

        inputs = {}
        outputs = {}
        service = {}

        for idx in range(len(self.dataframe[split_name])):
            augm_data = ''
            for inp_id in self.version_data.inputs.keys():
                tmp = []
                for col_name, data in self.instructions[inp_id].items():
                    dict_to_pass = data.copy()
                    if data['augmentation'] and split_name == 'train':
                        dict_to_pass.update([('augm_data', self.dataframe[split_name].iloc[idx, 1])])
                    sample = self.version_paths_data.sources.joinpath(self.dataframe[split_name].loc[idx, col_name])
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[inp_id][col_name]}, **dict_to_pass)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    if isinstance(array, tuple):
                        tmp.append(array[0])
                        augm_data += array[1]
                    else:
                        tmp.append(array)

                inputs[str(inp_id)] = np.concatenate(tmp, axis=0)

            for out_id in self.version_data.outputs.keys():
                for col_name, data in self.instructions[out_id].items():
                    tmp_im = Image.open(self.version_paths_data.sources.joinpath(self.dataframe[split_name].iloc[idx, 0]))
                    data.update([('orig_x', tmp_im.width),
                                 ('orig_y', tmp_im.height)])
                    if augm_data and split_name == 'train':
                        data_to_pass = augm_data
                    else:
                        data_to_pass = self.dataframe[split_name].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(data_to_pass, **{
                        'preprocess': self.preprocessing.preprocessing[out_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    for n in range(3):
                        outputs[str(out_id + n)] = np.array(array[n])
                        service[str(out_id + n)] = np.array(array[n+3])
            yield inputs, outputs, service

    # def keras_datasets(self):
    #
    #     (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()
    #
    #     if self.data.alias in ['mnist', 'fashion_mnist']:
    #         x_train = x_train[..., None]
    #         x_val = x_val[..., None]
    #     for out in self.data.outputs.keys():
    #         if self.data.outputs[out].task == LayerOutputTypeChoice.Classification:
    #             y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
    #             y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))
    #
    #     # for split in ['train', 'val']:
    #     #     for key in self.data.inputs.keys():
    #     #         self.X[split][str(key)] = globals()[f'x_{split}']
    #     #     for key in self.data.outputs.keys():
    #     #         self.Y[split][str(key)] = globals()[f'y_{split}']
    #     for key in self.data.inputs.keys():
    #         self.X['train'][str(key)] = x_train
    #         self.X['val'][str(key)] = x_val
    #     for key in self.data.outputs.keys():
    #         self.Y['train'][str(key)] = y_train
    #         self.Y['val'][str(key)] = y_val

    def version(self, alias: str):

        if self.dataset_data.group != DatasetGroupChoice.keras:
            parent_version_path = self.parent_dataset_paths_data.versions.joinpath('.'.join([alias, VERSION_EXT]))
            version_path = self.dataset_paths_data.versions.joinpath('.'.join([alias, VERSION_EXT]))
            if not version_path.is_dir():
                shutil.copytree(parent_version_path, version_path)
                with zipfile.ZipFile(version_path.joinpath('version.zip'), 'r') as z_file:
                    z_file.extractall(Path(version_path))
                version_path.joinpath('version.zip').unlink(missing_ok=True)
            self.version_paths_data = VersionPathsData(basepath=version_path)

            with open(self.version_paths_data.basepath.joinpath(VERSION_CONFIG), 'r') as ver_config:
                self.version_data = VersionData(**json.load(ver_config))

            self.preprocessing = CreatePreprocessing(dataset_path=self.version_paths_data.preprocessing)

            for split in self.dataframe:
                self.dataframe[split] = pd.read_csv(
                    self.version_paths_data.instructions.joinpath('tables', f'{split}.csv'),
                    index_col=0
                )

            for put_id in list(self.version_data.inputs) + list(self.version_data.outputs):
                self.instructions[put_id] = {}
                for instr_json in os.listdir(self.version_paths_data.instructions.joinpath('parameters')):
                    idx, *name = os.path.splitext(instr_json)[0].split('_')
                    name = '_'.join(name)
                    if put_id == int(idx):
                        with open(self.version_paths_data.instructions.joinpath('parameters', instr_json),
                                  'r') as instr:
                            self.instructions[put_id].update([(f'{idx}_{name}', json.load(instr))])

            if self.version_data.use_generator:
                copy_archive = False
                for col_name in self.instructions.values():
                    for param in col_name.values():
                        if param['put_type'] in ['image', 'segmentation']:
                            copy_archive = True
                if copy_archive:
                    shutil.copyfile(self.parent_dataset_paths_data.basepath.joinpath('sources.zip'),
                                    self.dataset_paths_data.basepath.joinpath('sources.zip'))
                    with zipfile.ZipFile(self.dataset_paths_data.basepath.joinpath('sources.zip'), 'r') as z_file:
                        z_file.extractall(self.version_paths_data.sources)
                    self.dataset_paths_data.basepath.joinpath('sources.zip').unlink()
        else:
            for version in VersionsGroup[0]['datasets'][0][self.dataset_data.alias]:
                if version['alias'] == alias:
                    self.version_data = VersionData(**version)

            # (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.dataset_data.alias).load_data()
            #
            # if self.dataset_data.alias in ['mnist', 'fashion_mnist'] and self.version_data.alias == 'add_dimension':
            #     x_train = x_train[..., None]
            #     x_val = x_val[..., None]
            # y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
            # y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))
            #
            # for key in self.version_data.inputs.keys():
            #     self.X['train'][str(key)] = x_train
            #     self.X['val'][str(key)] = x_val
            # for key in self.version_data.outputs.keys():
            #     self.Y['train'][str(key)] = y_train
            #     self.Y['val'][str(key)] = y_val

            # del x_train, y_train, x_val, y_val

        print(self.__str__())

        return self

    def prepare_dataset(self):

        if self.dataset_data.group == DatasetGroupChoice.keras:

            (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.dataset_data.alias).load_data()

            if self.dataset_data.alias in ['mnist', 'fashion_mnist'] and self.version_data.alias == 'add_dimension':
                x_train = x_train[..., None]
                x_val = x_val[..., None]
            y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
            y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))

            for key in self.version_data.inputs.keys():
                self.X['train'][str(key)] = x_train
                self.X['val'][str(key)] = x_val
            for key in self.version_data.outputs.keys():
                self.Y['train'][str(key)] = y_train
                self.Y['val'][str(key)] = y_val

            del x_train, y_train, x_val, y_val

            for put_id, data in KerasInstructions[self.dataset_data.alias].items():
                self.instructions[put_id] = data

            self.preprocessing.create_scaler(**{'put': 1, 'scaler': 'min_max_scaler',
                                                'min_scaler': 0, 'max_scaler': 1,
                                                'cols_names': f'1_{self.dataset_data.alias}'})
            self.preprocessing.preprocessing[1][f'1_{self.dataset_data.alias}'].fit(self.X['train']['1'].reshape(-1, 1))
            for key in self.X.keys():
                for inp in self.X[key]:
                    self.X[key][inp] = self.preprocessing.preprocessing[1][f'1_{self.dataset_data.alias}']\
                        .transform(self.X[key][inp].reshape(-1, 1)).reshape(self.X[key][inp].shape)

            for split in self.dataset:
                if self.service[split]:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                      self.Y[split],
                                                                      self.service[split]))
                else:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                      self.Y[split]))

        elif self.dataset_data.group in [DatasetGroupChoice.terra, DatasetGroupChoice.trds]:

            self.preprocessing.load_preprocesses(self.version_data.columns)

            if self.version_data.use_generator:
                num_inputs = len(self.version_data.inputs)
                num_outputs = len(self.version_data.outputs)
                if self.dataset_data.tags[num_inputs].alias == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    gen = self.generator_object_detection
                    out_signature = (
                        {str(x): tf.TensorSpec(shape=self.version_data.inputs[x].shape,
                                               dtype=self.version_data.inputs[x].dtype)
                         for x in range(1, num_inputs + 1)},
                        {str(x): tf.TensorSpec(shape=self.version_data.outputs[x].shape,
                                               dtype=self.version_data.outputs[x].dtype)
                         for x in range(num_inputs + 1, num_inputs + num_outputs + 1)},
                        {str(x): tf.TensorSpec(shape=self.version_data.service[x].shape,
                                               dtype=self.version_data.service[x].dtype)
                         for x in range(num_inputs + 1, num_inputs + num_outputs + 1)})
                else:
                    gen = self.generator_common
                    out_signature = (
                        {str(x): tf.TensorSpec(shape=self.version_data.inputs[x].shape,
                                               dtype=self.version_data.inputs[x].dtype)
                         for x in range(1, num_inputs + 1)},
                        {str(x): tf.TensorSpec(shape=self.version_data.outputs[x].shape,
                                               dtype=self.version_data.outputs[x].dtype)
                         for x in range(num_inputs + 1, num_outputs + num_inputs + 1)})

                self.dataset['train'] = Dataset.from_generator(lambda: gen(split_name='train'),
                                                               output_signature=out_signature)
                self.dataset['val'] = Dataset.from_generator(lambda: gen(split_name='val'),
                                                             output_signature=out_signature)
            else:
                for split in self.dataset:
                    for index in self.version_data.inputs.keys():
                        self.X[split][str(index)] = joblib.load(self.version_paths_data.arrays.joinpath(split,
                                                                                                        f'{index}.gz'))
                    for index in self.version_data.outputs.keys():
                        self.Y[split][str(index)] = joblib.load(self.version_paths_data.arrays.joinpath(split,
                                                                                                        f'{index}.gz'))
                    for index in self.version_data.service.keys():
                        if self.version_data.service[index]:
                            self.service[split][str(index)] = joblib.load(
                                self.version_paths_data.arrays.joinpath(split, f'{index}_service.gz')
                            )

                for split in ['train', 'val']:
                    if self.service[split]:
                        self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                          self.Y[split],
                                                                          self.service[split]))
                    else:
                        self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                          self.Y[split]))

        pass

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
