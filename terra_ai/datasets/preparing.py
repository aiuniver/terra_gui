import os
import numpy as np
import joblib
import json
import pandas as pd
import tensorflow as tf
import shutil
import zipfile

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from PIL import Image
from pathlib import Path

from terra_ai.utils import decamelize
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.data.datasets.dataset import DatasetData, DatasetPathsData, VersionData, VersionPathsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice
from terra_ai.data.presets.datasets import KerasInstructions
from terra_ai.settings import DATASET_CONFIG, VERSION_EXT, VERSION_CONFIG


class PrepareDataset(object):

    dataframe: dict = {'train': None, 'val': None}
    instructions: dict = {}
    X: dict = {'train': {}, 'val': {}}
    Y: dict = {'train': {}, 'val': {}}
    service: dict = {'train': {}, 'val': {}}
    dataset: dict = {'train': None, 'val': None}
    version_data = None
    preprocessing = None

    def __init__(self,
                 data: DatasetData,
                 datasets_path: Path = Path(''),
                 parent_datasets_path: Path = Path('')
                 ):

        self.dataset_data: DatasetData = data
        if self.dataset_data.group != DatasetGroupChoice.keras:
            self.version_paths_data = None
            self.dataset_paths_data = DatasetPathsData(basepath=datasets_path)
            self.parent_dataset_paths_data = DatasetPathsData(basepath=parent_datasets_path)

        # else:
        #     self.preprocessing = CreatePreprocessing()

    def __str__(self):

        version = f'{self.version_data.alias}/{self.version_data.name}' if self.version_data else "не выбрана"

        return f'Датасет: {self.dataset_data.alias}/{self.dataset_data.name}.\n' \
               f'Версия: {version}.'

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
            """
            Тут всякие движения для керасовских датасетов
            """

        print(self.__str__())

        return self

    def prepare_dataset(self):

        if self.dataset_data.group == DatasetGroupChoice.keras:

            self.keras_datasets()

            for put_id, data in KerasInstructions[self.dataset_data.alias].items():
                self.instructions[put_id] = data

            self.preprocessing.create_scaler(**{'put': 1, 'scaler': 'min_max_scaler',
                                                'min_scaler': 0, 'max_scaler': 1,
                                                'cols_names': f'1_{self.data.alias}'})
            self.preprocessing.preprocessing[1][f'1_{self.data.alias}'].fit(self.X['train']['1'].reshape(-1, 1))
            for key in self.X.keys():
                for inp in self.X[key]:
                    self.X[key][inp] = self.preprocessing.preprocessing[1][f'1_{self.data.alias}']\
                        .transform(self.X[key][inp].reshape(-1, 1)).reshape(self.X[key][inp].shape)

            for split in self.dataset:
                if self.service[split]:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                      self.Y[split],
                                                                      self.service[split]))
                else:
                    self.dataset[split] = Dataset.from_tensor_slices((self.X[split],
                                                                      self.Y[split]))

        elif self.dataset_data.group in [DatasetGroupChoice.terra, DatasetGroupChoice.custom]:

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
