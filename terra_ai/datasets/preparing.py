import os
import numpy as np
import joblib
import json
import pandas as pd

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tensorflow.python.ops.ragged.ragged_factory_ops import constant
from sklearn.model_selection import train_test_split

from terra_ai.utils import decamelize
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.datasets.dataset import DatasetPathsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.datasets.arrays_create import CreateArray


class PrepareDataset(object):

    def __init__(self, data: DatasetData, datasets_path=None):

        self.data = data
        self.language = None
        self.instructions: dict = {'inputs': {}, 'outputs': {}}
        self.dts_prepared: bool = False
        self.dataframe: dict = {}
        if self.data.group != 'keras':
            self.paths = DatasetPathsData(basepath=datasets_path)
            self.preprocessing = CreatePreprocessing(dataset_path=self.paths.basepath)
        else:
            self.preprocessing = CreatePreprocessing()

        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.dataset: dict = {}

        pass

    def train_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['train'])):
            for key, value in self.data.inputs.items():
                if value.task == LayerInputTypeChoice.Text:
                    sample = self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['inputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                inputs[str(key)] = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                                  **arr['parameters'])

            for key, value in self.data.outputs.items():
                if value.task in [LayerOutputTypeChoice.Text, LayerOutputTypeChoice.TextSegmentation]:
                    sample = self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['outputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                       **arr['parameters'])
                if value.task == LayerOutputTypeChoice.ObjectDetection:
                    for m in range(len(array)):
                        outputs[str(key + m)] = array[m]
                    break
                else:
                    outputs[str(key)] = array

            yield inputs, outputs

    def val_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['val'])):
            for key, value in self.data.inputs.items():
                if value.task == LayerInputTypeChoice.Text:
                    sample = self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['inputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                inputs[str(key)] = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                                  **arr['parameters'])

            for key, value in self.data.outputs.items():
                if value.task in [LayerOutputTypeChoice.Text, LayerOutputTypeChoice.TextSegmentation]:
                    sample = self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['outputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                       **arr['parameters'])
                if value.task == LayerOutputTypeChoice.ObjectDetection:
                    for m in range(len(array)):
                        outputs[str(key + m)] = array[m]
                    break
                else:
                    outputs[str(key)] = array

            yield inputs, outputs

    def test_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['test'])):
            for key, value in self.data.inputs.items():

                if value.task == LayerInputTypeChoice.Text:
                    sample = self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['inputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                inputs[str(key)] = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                                  **arr['parameters'])

            for key, value in self.data.outputs.items():
                if value.task in [LayerOutputTypeChoice.Text, LayerOutputTypeChoice.TextSegmentation]:
                    sample = self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}']
                else:
                    sample = os.path.join(self.paths.basepath,
                                          self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'])

                arr = getattr(CreateArray(), f'create_{decamelize(value.task)}')(
                    sample,
                    **self.instructions['outputs'][key],
                    **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{decamelize(value.task)}')(arr['instructions'],
                                                                                       **arr['parameters'])
                if value.task == LayerOutputTypeChoice.ObjectDetection:
                    for m in range(len(array)):
                        outputs[str(key + m)] = array[m]
                    break
                else:
                    outputs[str(key)] = array

            yield inputs, outputs

    def keras_datasets(self):

        (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()

        if self.data.alias in ['mnist', 'fashion_mnist']:
            x_train = x_train[..., None]
            x_val = x_val[..., None]
        for out in self.data.outputs.keys():
            # for col_name, data in self.data.outputs[out].items():
            if self.data.outputs[out].task == LayerOutputTypeChoice.Classification:
                y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
                y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))

        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=True)
        self.X['train']['1'] = x_train
        self.X['val']['1'] = x_val
        self.X['test']['1'] = x_test
        self.Y['train']['2'] = y_train
        self.Y['val']['2'] = y_val
        self.Y['test']['2'] = y_test

        pass

    def prepare_dataset(self):

        def load_arrays():

            for sample in os.listdir(self.paths.arrays):
                for index in self.data.inputs.keys():
                    self.X[sample][str(index)] = joblib.load(os.path.join(self.paths.arrays, sample, f'{index}.gz'))
                for index in self.data.outputs.keys():
                    self.Y[sample][str(index)] = joblib.load(os.path.join(self.paths.arrays, sample, f'{index}.gz'))

            pass

        if self.data.group == DatasetGroupChoice.keras:

            self.keras_datasets()

            if self.data.alias in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.preprocessing.create_scaler(array=self.X['train']['1'], **{'put': 1, 'scaler': 'min_max_scaler',
                                                                                'min_scaler': 0, 'max_scaler': 1,
                                                                                'cols_names': f'1_{self.data.alias}'})
                for key in self.X.keys():
                    for inp in self.X[key]:
                        self.X[key][inp] = self.preprocessing.preprocessing[1][f'1_{self.data.alias}']\
                            .transform(self.X[key][inp].reshape(-1, 1)).reshape(self.X[key][inp].shape)

            if self.data.alias in ['imdb', 'reuters']:
                for key in self.X.keys():
                    for inp in self.X[key]:
                        self.X[key][inp] = constant(self.X[key][inp])

            self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
            self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
            self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        elif self.data.group in [DatasetGroupChoice.terra, DatasetGroupChoice.custom]:
            for put in ['train', 'val', 'test']:
                self.dataframe[put] = pd.read_csv(os.path.join(self.paths.instructions, 'tables', f'{put}.csv'),
                                                  index_col=0)

            self.preprocessing.load_preprocesses(self.data.columns)

            if self.data.use_generator:
                for instr in os.listdir(os.path.join(self.paths.instructions, 'parameters')):
                    with open(os.path.join(self.paths.instructions, 'parameters', instr), 'r') as instruction:
                        ins = json.load(instruction)
                        instr = instr[:instr.rfind('.')]
                        idx, put = instr.split('_')
                        self.instructions[put][int(idx)] = getattr(CreateArray(),
                                                                   f'instructions_{decamelize(ins["type"])}')([], **ins)['parameters']

                # self.dataset['train'] = Dataset.from_generator(self.train_generator,
                #                                                output_shapes=(prepare_generator_parameters(
                #                                                    self.data.inputs, 'shapes'),
                #                                                               prepare_generator_parameters(
                #                                                                   self.data.outputs, 'shapes')),
                #                                                output_types=(prepare_generator_parameters(
                #                                                    self.data.inputs, 'types'),
                #                                                              prepare_generator_parameters(
                #                                                                  self.data.outputs, 'types'))
                #                                                )
                # self.dataset['val'] = Dataset.from_generator(self.val_generator,
                #                                              output_shapes=(prepare_generator_parameters(
                #                                                  self.data.inputs, 'shapes'),
                #                                                             prepare_generator_parameters(
                #                                                                 self.data.outputs, 'shapes')),
                #                                              output_types=(prepare_generator_parameters(
                #                                                  self.data.inputs, 'types'),
                #                                                            prepare_generator_parameters(
                #                                                                self.data.outputs, 'types'))
                #                                              )
                # self.dataset['test'] = Dataset.from_generator(self.test_generator,
                #                                               output_shapes=(prepare_generator_parameters(
                #                                                   self.data.inputs, 'shapes'),
                #                                                              prepare_generator_parameters(
                #                                                                  self.data.outputs, 'shapes')),
                #                                               output_types=(prepare_generator_parameters(
                #                                                   self.data.inputs, 'types'),
                #                                                             prepare_generator_parameters(
                #                                                                 self.data.outputs, 'types'))
                #                                               )
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        self.dts_prepared = True

        pass
