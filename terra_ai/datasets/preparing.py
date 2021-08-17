import os
import numpy as np
import joblib
import json
import pandas as pd

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras import utils
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from terra_ai.data.datasets.creation import SourceData
from . import arrays_create, array_creator
from . import loading as datasets_loading
from .data import Preprocesses
from ..data.datasets.dataset import DatasetData, DatasetLoadData
from ..data.datasets.extra import DatasetGroupChoice, SourceModeChoice


class PrepareDTS(object):

    def __init__(self, zip_params, trds_path='/content/drive/MyDrive/TerraAI/datasets'):

        self.name: str = ''
        self.source: str = ''
        self.language = None
        self.trds_path: str = trds_path
        self.source_path: str = ''
        self.inputs: dict = {}
        self.input_shape: dict = {}
        self.input_dtype: dict = {}
        self.input_datatype: str = ''
        self.input_names: dict = {}
        self.outputs: dict = {}
        self.output_shape: dict = {}
        self.output_dtype: dict = {}
        self.output_datatype: dict = {}
        self.output_names: dict = {}
        self.split_sequence: dict = {}
        self.file_folder: str = ''
        self.use_generator: bool = False
        self.zip_params: dict = {}
        self.instructions: dict = {'inputs': {}, 'outputs': {}}
        self.tags: dict = {}
        self.put_tags: dict = {'inputs': {}, 'outputs': {}}
        self.task_type: dict = {}
        self.one_hot_encoding: dict = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.dts_prepared: bool = False
        self.dataframe: dict = {}
        self.zip_params = zip_params

        self.dataloader = None
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
            for key, value in self.put_tags['inputs'].items():
                inputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['train'].loc[idx, f'{key}_{value}'], **self.instructions['inputs'][int(key)])
            for key, value in self.put_tags['outputs'].items():
                if 'object_detection' in self.put_tags['outputs'].values():
                    arrays = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['train'].loc[idx, f'2_{value}'], **self.instructions['outputs'][int(key)])
                    for i in range(6):
                        outputs[int(key) + i] = np.array(arrays[i])
                else:
                    outputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['train'].loc[idx, f'{key}_{value}'], **self.instructions['outputs'][int(key)])

            yield inputs, outputs

    def val_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['val'])):
            for key, value in self.put_tags['inputs'].items():
                inputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['val'].loc[idx, f'{key}_{value}'], **self.instructions['inputs'][int(key)])
            for key, value in self.put_tags['outputs'].items():
                if 'object_detection' in self.put_tags['outputs'].values():
                    arrays = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['val'].loc[idx, f'2_{value}'], **self.instructions['outputs'][int(key)])
                    for i in range(6):
                        outputs[int(key) + i] = np.array(arrays[i])
                else:
                    outputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['val'].loc[idx, f'{key}_{value}'], **self.instructions['outputs'][int(key)])

            yield inputs, outputs

    def test_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['test'])):
            for key, value in self.put_tags['inputs'].items():
                inputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['test'].loc[idx, f'{key}_{value}'], **self.instructions['inputs'][int(key)])
            for key, value in self.put_tags['outputs'].items():
                if 'object_detection' in self.put_tags['outputs'].values():
                    arrays = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['test'].loc[idx, f'2_{value}'], **self.instructions['outputs'][int(key)])
                    for i in range(6):
                        outputs[int(key) + i] = np.array(arrays[i])
                else:
                    outputs[int(key)] = getattr(self.createarray, f"create_{value}")(self.source_path, self.dataframe['test'].loc[idx, f'{key}_{value}'], **self.instructions['outputs'][int(key)])

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

    def prepare_dataset(self, dataset_data: DatasetLoadData):

        def load_arrays():

            # for sample in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays')):
            #     for arr in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample)):
            #         put_id = int(arr[:arr.rfind('.')])
            #         if str(put_id) in self.put_tags['inputs'].keys():   #list(dataset_data.inputs.keys()):
            #             self.X[sample][put_id] = joblib.load(
            #                 os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, arr))
            #         elif str(put_id) in self.put_tags['outputs'].keys():   #list(dataset_data.outputs.keys()):
            #             if 'object_detection' in self.put_tags['outputs'][str(put_id)]:
            #                 for i in range(6):
            #                     self.Y[sample][put_id+i] = joblib.load(
            #                         os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, arr))
            #             else:
            #                 self.Y[sample][put_id] = joblib.load(
            #                     os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, arr))

            for sample in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays')):
                for index, tag in self.put_tags['inputs'].items():
                    self.X[sample][int(index)] = joblib.load(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, f'{index}.gz'))
                for index, tag in self.put_tags['outputs'].items():
                    if tag == 'object_detection':
                        for i in range(6):
                            self.Y[sample][int(index)+i] = joblib.load(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, f'{index+i}.gz'))
                    else:
                        self.Y[sample][int(index)] = joblib.load(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'arrays', sample, f'{index}.gz'))

            pass

        def load_scalers(puts):
            scalers = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'scalers')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    scalers.append(arr[:-3])
            for put in puts:
                if put in scalers:
                    self.createarray.scaler[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.scaler[put] = None

            pass

        def load_tokenizer(puts):
            tokenizer = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'tokenizer')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    tokenizer.append(arr[:-3])

            for put in puts:
                if put in tokenizer:
                    self.createarray.tokenizer[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.tokenizer[put] = None

            pass

        def load_word2vec(puts):

            word2v = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'word2vec')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    word2v.append(arr[:-3])

            for put in puts:
                if put in word2v:
                    self.createarray.word2vec[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.word2vec[put] = None

            pass

        def load_augmentation(puts):

            augmentation = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'augmentation')
            if os.path.exists(folder_path):
                for aug in os.listdir(folder_path):
                    augmentation.append(aug[:-3])

            for put in puts:
                if put in augmentation:
                    self.createarray.augmentation[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.augmentation[put] = None

            pass

        if dataset_data.group == DatasetGroupChoice.keras and dataset_data.name in \
                ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters']:
            if dataset_data.name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(dataset_data.name, one_hot_encoding=True, scaler='MinMaxScaler', net='conv')
                self.task_type['output_1'] = 'classification'
            elif dataset_data.name == 'imdb':
                self.keras_datasets(dataset_data.name, one_hot_encoding=True)
                self.task_type['output_1'] = 'classification'
            elif dataset_data.name == 'reuters':
                self.keras_datasets(dataset_data.name)
                self.task_type['output_1'] = 'classification'
            elif dataset_data.name == 'boston_housing':
                self.keras_datasets(dataset_data.name, scaler='StandardScaler')
                self.task_type['output_1'] = 'regression'
        elif dataset_data.group == DatasetGroupChoice.custom:

            self.trds_path = os.path.join(os.path.sep, 'content', 'drive', 'MyDrive', 'TerraAI', 'datasets')
            with open(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'config.json'), 'r') as config:
                data = json.load(config)
            for key, value in data.items():
                self.__dict__[key] = value

            inp_datatype = ''
            for key in self.inputs.keys():
                self.input_names[int(key)] = self.inputs[key]['name']
                self.input_shape[int(key)] = self.inputs[key]['shape']
                self.input_datatype = ' '.join([inp_datatype, self.inputs[key]['datatype']])
                self.input_dtype[int(key)] = self.inputs[key]['dtype']
            for key in self.outputs.keys():
                self.output_names[int(key)] = self.outputs[key]['name']
                self.output_shape[int(key)] = self.outputs[key]['shape']
                self.output_datatype[int(key)] = self.outputs[key]['datatype']
                self.output_dtype[int(key)] = self.outputs[key]['dtype']
                self.task_type[int(key)] = self.outputs[key]['task']

            for put in ['train', 'val', 'test']:
                self.dataframe[put] = pd.read_csv(os.path.join(self.trds_path, f'dataset {dataset_data.name}',
                                                               'instructions', f'{put}.csv'), index_col=0)
            if self.use_generator:
                for instr in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'instructions')):
                    if instr.endswith('.json'):
                        with open(os.path.join(self.trds_path, f'dataset {dataset_data.name}', 'instructions', instr), 'r') as instruction:
                            ins = json.load(instruction)
                        instr = instr[:instr.rfind('.')]
                        idx, put = instr.split('_')
                        self.instructions[put][int(idx)] = ins
                datasets_loading.source(strict_object=SourceData(**self.zip_params))
                self.source_path = '/tmp/terraai/datasets_sources/googledrive/chess'  # self.dataloader.file_folder

                self.dataset['train'] = Dataset.from_generator(self.train_generator,
                                                               output_shapes=(self.input_shape, self.output_shape),
                                                               output_types=(self.input_dtype, self.output_dtype)
                                                               )
                self.dataset['val'] = Dataset.from_generator(self.val_generator,
                                                             output_shapes=(self.input_shape, self.output_shape),
                                                             output_types=(self.input_dtype, self.output_dtype)
                                                             )
                self.dataset['test'] = Dataset.from_generator(self.test_generator,
                                                              output_shapes=(self.input_shape, self.output_shape),
                                                              output_types=(self.input_dtype, self.output_dtype)
                                                              )
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        # load_scalers(puts)
        # load_tokenizer(puts)
        # load_word2vec(puts)
        # load_augmentation(puts)
        # self.load_preprocesses(dataset_data, puts)

        self.dts_prepared = True

        pass

    def load_preprocesses(self, dataset_data: DatasetData, puts: list):
        for preprocess_name in Preprocesses:
            preprocess = getattr(array_creator, preprocess_name)
            preprocess_data = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_data.name}', preprocess_name)
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    preprocess_data.append(int(arr[:-3]))

            for put in puts:
                if put in preprocess_data:
                    preprocess[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    preprocess[put] = None
