import os
import numpy as np
import joblib
import json

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100, imdb, reuters, boston_housing
from tensorflow.keras import utils
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from terra_ai.data.datasets.creation import SourceData
from . import arrays_create
from . import loading as datasets_loading


class PrepareDTS(object):

    def __init__(self, trds_path='/content/drive/MyDrive/TerraAI/datasets'):

        self.name: str = ''
        self.source: str = ''
        self.language = None
        self.trds_path: str = trds_path
        self.input_shape: dict = {}
        self.input_dtype: dict = {}
        self.input_datatype: str = ''
        self.input_names: dict = {}
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
        self.task_type: dict = {}
        self.one_hot_encoding: dict = {}
        self.num_classes: dict = {}
        self.classes_names: dict = {}
        self.classes_colors: dict = {}
        self.dts_prepared: bool = False

        self.dataloader = None
        self.createarray = arrays_create

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

    def generator_train(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['train']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(3):
                        outputs[f'output_{int(key[-1])+i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

            yield inputs, outputs

    def generator_val(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['val']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(3):
                        outputs[f'output_{int(key[-1])+i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

            yield inputs, outputs

    def generator_test(self):

        inputs = {}
        outputs = {}
        for idx in self.split_sequence['test']:
            for key in self.instructions['inputs'].keys():
                inputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                    self.instructions['inputs'][key]['instructions'][idx],
                    **self.instructions['inputs'][key]['parameters'])
            for key in self.instructions['outputs'].keys():
                if 'object_detection' in self.tags.values():
                    arrays = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])
                    for i in range(3):
                        outputs[f'output_{int(key[-1])+i}'] = np.array(arrays[i])
                else:
                    outputs[key] = getattr(self.createarray, f"create_{self.tags[key]}")(
                        self.instructions['outputs'][key]['instructions'][idx],
                        **self.instructions['outputs'][key]['parameters'])

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

        if 'scaler' in options.keys() and options['scaler'] == 'MinMaxScaler' or \
                'scaler' in options.keys() and options['scaler'] == 'StandardScaler':

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

    def prepare_dataset(self, dataset_name: str, source: str):

        def load_arrays():

            for sample in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays')):
                for arr in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample)):
                    if 'input' in arr:
                        self.X[sample][arr[:arr.rfind('.')]] = joblib.load(
                            os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample, arr))
                    elif 'output' in arr:
                        self.Y[sample][arr[:arr.rfind('.')]] = joblib.load(
                            os.path.join(self.trds_path, f'dataset {dataset_name}', 'arrays', sample, arr))

            pass

        def load_scalers():

            scalers = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'scalers')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    scalers.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in scalers:
                    self.createarray.scaler[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.scaler[put] = None

            pass

        def load_tokenizer():

            tokenizer = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'tokenizer')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    tokenizer.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in tokenizer:
                    self.createarray.tokenizer[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.tokenizer[put] = None

            pass

        def load_word2vec():

            word2v = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'word2vec')
            if os.path.exists(folder_path):
                for arr in os.listdir(folder_path):
                    word2v.append(arr[:-3])

            for put in list(self.tags.keys()):
                if put in word2v:
                    self.createarray.word2vec[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.word2vec[put] = None

            pass

        def load_augmentation():

            augmentation = []
            folder_path = os.path.join(self.trds_path, f'dataset {dataset_name}', 'augmentation')
            if os.path.exists(folder_path):
                for aug in os.listdir(folder_path):
                    augmentation.append(aug[:-3])

            for put in list(self.tags.keys()):
                if put in augmentation:
                    self.createarray.augmentation[put] = joblib.load(os.path.join(folder_path, f'{put}.gz'))
                else:
                    self.createarray.augmentation[put] = None

            pass

        if dataset_name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing', 'reuters'] and \
                source != 'custom_dataset':
            if dataset_name in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.keras_datasets(dataset_name, one_hot_encoding=True, scaler='MinMaxScaler', net='conv')
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'imdb':
                self.keras_datasets(dataset_name, one_hot_encoding=True)
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'reuters':
                self.keras_datasets(dataset_name)
                self.task_type['output_1'] = 'classification'
            elif dataset_name == 'boston_housing':
                self.keras_datasets(dataset_name, scaler='StandardScaler')
                self.task_type['output_1'] = 'regression'
        elif source == 'custom_dataset':
            with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'config.json'), 'r') as cfg:
                data = json.load(cfg)
            for key, value in data.items():
                self.__dict__[key] = value
            if self.use_generator:
                if 'text' in self.tags.values():
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'txt_list.json'),
                              'r') as txt:
                        self.createarray.txt_list = json.load(txt)

                datasets_loading.load(strict_object=SourceData(**self.zip_params))

                with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'sequence.json'),
                          'r') as cfg:
                    self.split_sequence = json.load(cfg)
                for inp in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions',
                                                   'inputs')):
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'inputs', inp),
                              'r') as cfg:
                        data = json.load(cfg)
                    self.instructions['inputs'][inp[:inp.rfind('.')]] = data
                for out in os.listdir(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions',
                                                   'outputs')):
                    with open(os.path.join(self.trds_path, f'dataset {dataset_name}', 'instructions', 'outputs', out),
                              'r') as cfg:
                        data = json.load(cfg)
                    self.instructions['outputs'][out[:out.rfind('.')]] = data
                self.createarray.file_folder = self.dataloader.file_folder

                self.dataset['train'] = Dataset.from_generator(self.generator_train,
                                                               output_shapes=(self.input_shape, self.output_shape),
                                                               output_types=(self.input_dtype, self.output_dtype))
                self.dataset['val'] = Dataset.from_generator(self.generator_val,
                                                             output_shapes=(self.input_shape, self.output_shape),
                                                             output_types=(self.input_dtype, self.output_dtype))
                self.dataset['test'] = Dataset.from_generator(self.generator_test,
                                                              output_shapes=(self.input_shape, self.output_shape),
                                                              output_types=(self.input_dtype, self.output_dtype))
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        load_scalers()
        load_tokenizer()
        load_word2vec()
        load_augmentation()

        self.dts_prepared = True

        pass
