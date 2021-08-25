import os
import numpy as np
import joblib
import json
import pandas as pd
from ..utils import decamelize

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras import utils
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from . import array_creator
from .data import Preprocesses
from ..data.datasets.dataset import DatasetData
from ..data.datasets.extra import DatasetGroupChoice


class PrepareDTS(object):

    def __init__(self, data: DatasetData):

        self.data = data
        self.language = None
        self.instructions: dict = {'inputs': {}, 'outputs': {}}
        self.dts_prepared: bool = False
        self.dataframe: dict = {}

        self.createarray = array_creator
        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.dataset: dict = {}

        pass

    @staticmethod
    def _set_datatype(shape) -> str:

        datatype = {0: 'DIM',
                    1: 'DIM',
                    2: 'DIM',
                    3: '1D',
                    4: '2D',
                    5: '3D'
                    }

        return datatype[len(shape)]

    @staticmethod
    def _set_language(name: str):

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

    def train_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['train'])):
            for key, value in self.data.inputs.items():
                inputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['train'].loc[idx, f'2_{decamelize(value.task)}'], **self.instructions['outputs'][2])
                    outputs[key] = np.array(arrays[key-2])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['outputs'][key])

            yield inputs, outputs

    def val_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['val'])):
            for key, value in self.data.inputs.items():
                inputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['val'].loc[idx, f'2_{decamelize(value.task)}'], **self.instructions['outputs'][2])
                    outputs[key] = np.array(arrays[key-2])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['outputs'][key])

            yield inputs, outputs

    def test_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['test'])):
            for key, value in self.data.inputs.items():
                inputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['test'].loc[idx, f'2_{decamelize(value.task)}'], **self.instructions['outputs'][2])
                    outputs[key] = np.array(arrays[key-2])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{decamelize(value.task)}")(self.data.paths.dataset_sources, self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'], **self.instructions['outputs'][key])

            yield inputs, outputs

    def keras_datasets(self, dataset: str, **options):

        self.name = dataset.lower()
        tags = {'mnist': {'input_1': 'images', 'output_1': 'classification'},
                'fashion_mnist': {'input_1': 'images', 'output_1': 'classification'},
                'cifar10': {'input_1': 'images', 'output_1': 'classification'},
                'cifar100': {'input_1': 'images', 'output_1': 'classification'},
                'imdb': {'input_1': 'text', 'output_1': 'classification'},
                'boston_housing': {'input_1': 'text', 'output_1': 'regression'},
                'reuters': {'input_1': 'text', 'output_1': 'classification'}}
        self.tags = tags[self.name]
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
        (x_train, y_train), (x_val, y_val) = data[self.name].load_data()

        self.language = self._set_language(self.name)
        if 'classification' in self.tags['output_1']:
            self.num_classes['output_1'] = len(np.unique(y_train, axis=0))
            if self.name == 'fashion_mnist':
                self.classes_names['output_1'] = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                                  'Shirt',
                                                  'Sneaker', 'Bag', 'Ankle boot']
            elif self.name == 'cifar10':
                self.classes_names['output_1'] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                                                  'horse', 'ship',
                                                  'truck']
            else:
                self.classes_names['output_1'] = [str(i) for i in range(len(np.unique(y_train, axis=0)))]
        else:
            self.num_classes['output_1'] = 1

        if 'net' in options.keys() and self.name in list(data.keys())[:4]:
            if options['net'].lower() == 'linear':
                x_train = x_train.reshape((-1, np.prod(np.array(x_train.shape)[1:])))
                x_val = x_val.reshape((-1, np.prod(np.array(x_val.shape)[1:])))
            elif options['net'].lower() == 'conv':
                if len(x_train.shape) == 3:
                    x_train = x_train[..., None]
                    x_val = x_val[..., None]

        if 'scaler' in options.keys() and options['scaler'] in ['MinMaxScaler', 'StandardScaler']:

            shape_xt = x_train.shape
            shape_xv = x_val.shape
            x_train = x_train.reshape(-1, 1)
            x_val = x_val.reshape(-1, 1)

            if options['scaler'] == 'MinMaxScaler':
                self.createarray.scaler['input_1'] = MinMaxScaler()
                if 'classification' not in self.tags['output_1']:
                    self.createarray.scaler['output_1'] = MinMaxScaler()

            elif options['scaler'] == 'StandardScaler':
                self.createarray.scaler['input_1'] = StandardScaler()
                if 'classification' not in self.tags['output_1']:
                    self.createarray.scaler['output_1'] = StandardScaler()

            self.createarray.scaler['input_1'].fit(x_train)
            x_train = self.createarray.scaler['input_1'].transform(x_train)
            x_val = self.createarray.scaler['input_1'].transform(x_val)
            x_train = x_train.reshape(shape_xt)
            x_val = x_val.reshape(shape_xv)

            if 'classification' not in self.tags['output_1']:
                shape_yt = y_train.shape
                shape_yv = y_val.shape
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
                self.createarray.scaler['output_1'].fit(y_train)
                y_train = self.createarray.scaler['output_1'].transform(y_train)
                y_val = self.createarray.scaler['output_1'].transform(y_val)
                y_train = y_train.reshape(shape_yt)
                y_val = y_val.reshape(shape_yv)
        else:
            self.createarray.scaler['output_1'] = None

        self.one_hot_encoding['output_1'] = False
        if 'one_hot_encoding' in options.keys() and options['one_hot_encoding'] is True:
            if 'classification' in self.tags['output_1']:
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))
                self.one_hot_encoding['output_1'] = True

        self.input_shape['input_1'] = x_train.shape if len(x_train.shape) < 2 else x_train.shape[1:]
        self.input_datatype = self._set_datatype(shape=x_train.shape)
        self.input_names['input_1'] = 'Вход'
        self.output_shape['output_1'] = y_train.shape[1:]
        self.output_datatype['output_1'] = self._set_datatype(shape=y_train.shape)
        self.output_names['input_1'] = 'Выход'

        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=True)
        self.X['train']['input_1'] = x_train
        self.X['val']['input_1'] = x_val
        self.X['test']['input_1'] = x_test
        self.Y['train']['output_1'] = y_train
        self.Y['val']['output_1'] = y_val
        self.Y['test']['output_1'] = y_test

        self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
        self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
        self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        return self

    def prepare_dataset(self):

        def load_arrays():

            for sample in os.listdir(self.data.paths.arrays):
                for index in self.data.inputs.keys():
                    self.X[sample][index] = joblib.load(os.path.join(self.data.paths.arrays, sample, f'{index}.gz'))
                for index, data in self.data.outputs.items():
                    if data.task == 'ObjectDetection':
                        for i in range(6):
                            self.Y[sample][index] = joblib.load(os.path.join(self.data.paths.arrays, sample, f'{index}.gz'))
                    else:
                        self.Y[sample][index] = joblib.load(os.path.join(self.data.paths.arrays, sample, f'{index}.gz'))

            pass

        def load_preprocess(parameter):

            sample_list = []
            folder_path = os.path.join(self.data.paths.datasets, parameter)
            if os.path.exists(folder_path):
                for sample in os.listdir(folder_path):
                    sample_list.append(int(os.path.splitext(sample)[0]))

            for key in self.data.inputs.keys():
                if key in sample_list:
                    self.createarray.__dict__[parameter][key] = joblib.load(os.path.join(folder_path, f'{key}.gz'))
                else:
                    self.createarray.__dict__[parameter][key] = None

            pass

        if self.data.group == DatasetGroupChoice.keras and self.data.alias in \
                ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters']:
            if self.data.alias in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(self.data.alias, one_hot_encoding=True, scaler='MinMaxScaler', net='conv')
                # self.task_type['output_1'] = 'classification'
            elif self.data.alias == 'imdb':
                self.keras_datasets(self.data.alias, one_hot_encoding=True)
                # self.task_type['output_1'] = 'classification'
            elif self.data.alias == 'reuters':
                self.keras_datasets(self.data.alias)
                # self.task_type['output_1'] = 'classification'
            elif self.data.alias == 'boston_housing':
                self.keras_datasets(self.data.alias, scaler='StandardScaler')
                # self.task_type['output_1'] = 'regression'
        elif self.data.group == DatasetGroupChoice.custom:

            for put in ['train', 'val', 'test']:
                self.dataframe[put] = pd.read_csv(os.path.join(self.data.paths.instructions, 'tables', f'{put}.csv'), index_col=0)
            if self.data.use_generator:
                for instr in os.listdir(os.path.join(self.data.paths.instructions, 'parameters')):
                    with open(os.path.join(self.data.paths.instructions, 'parameters', instr), 'r') as instruction:
                        ins = json.load(instruction)
                    instr = instr[:instr.rfind('.')]
                    idx, put = instr.split('_')
                    self.instructions[put][int(idx)] = ins

                num_inputs = len(self.data.inputs)
                num_outputs = len(self.data.outputs)
                self.dataset['train'] = Dataset.from_generator(self.train_generator,
                                                               output_shapes=({x: self.data.inputs[x].shape for x in
                                                                               range(1, num_inputs+1)},
                                                                              {x: self.data.outputs[x].shape for x in
                                                                               range(num_inputs + 1, num_outputs + 2)}),
                                                               output_types=({x: self.data.inputs[x].dtype for x in
                                                                              range(1, num_inputs + 1)},
                                                                             {x: self.data.outputs[x].dtype for x in
                                                                              range(num_inputs + 1, num_outputs + 2)})
                                                               )
                self.dataset['val'] = Dataset.from_generator(self.val_generator,
                                                             output_shapes=({x: self.data.inputs[x].shape for x in
                                                                             range(1, num_inputs + 1)},
                                                                            {x: self.data.outputs[x].shape for x in
                                                                             range(num_inputs + 1, num_outputs + 2)}),
                                                             output_types=({x: self.data.inputs[x].dtype for x in
                                                                            range(1, num_inputs + 1)},
                                                                           {x: self.data.outputs[x].dtype for x in
                                                                            range(num_inputs + 1, num_outputs + 2)})
                                                             )
                self.dataset['test'] = Dataset.from_generator(self.test_generator,
                                                              output_shapes=({x: self.data.inputs[x].shape for x in
                                                                              range(1, num_inputs + 1)},
                                                                             {x: self.data.outputs[x].shape for x in
                                                                              range(num_inputs + 1, num_outputs + 2)}),
                                                              output_types=({x: self.data.inputs[x].dtype for x in
                                                                             range(1, num_inputs + 1)},
                                                                            {x: self.data.outputs[x].dtype for x in
                                                                             range(num_inputs + 1, num_outputs + 2)})
                                                              )
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        for param in ['scaler', 'tokenizer', 'word2vec', 'augmentation']:
            load_preprocess(param)

        self.dts_prepared = True

        pass

    def load_preprocesses(self, dataset_data: DatasetData, puts: list):
        for preprocess_name in Preprocesses:
            preprocess = getattr(array_creator, preprocess_name)
            preprocess_data = []
            folder_path = os.path.join(self.data.paths.datasets, preprocess_name)
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    preprocess_data.append(int(os.path.splitext(arr)[0]))

            for put in puts:
                if put in preprocess_data:
                    preprocess[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    preprocess[put] = None
