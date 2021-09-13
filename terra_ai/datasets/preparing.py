import os
import numpy as np
import joblib
import json
import pandas as pd
from ..utils import decamelize

from tensorflow.keras import utils
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from sklearn.model_selection import train_test_split

from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.datasets.dataset import DatasetPathsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice
from terra_ai.datasets.arrays_create import CreateArray
from tensorflow.keras import datasets as load_keras_datasets


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

        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.dataset: dict = {}

        pass

    # [@staticmethod]
    # def _set_language(name: str):

    #     language = {'imdb': 'English',
    #                 'boston_housing': 'English',
    #                 'reuters': 'English',
    #                 'заболевания': 'Russian',
    #                 'договоры': 'Russian',
    #                 'умный_дом': 'Russian',
    #                 'квартиры': 'Russian'
    #                 }

    #     if name in language.keys():
    #         return language[name]
    #     else:
    #         return None

    def train_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['train'])):
            for key, value in self.data.inputs.items():
                inputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                    self.paths.basepath,
                    self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'],
                    **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(self.paths.basepath, f"create_{decamelize(value.task)}")(
                        self.paths.basepath,
                        self.dataframe['train'].loc[idx, f'2_{decamelize(value.task)}'],
                        **self.instructions['outputs'][2])
                    outputs[str(key)] = np.array(arrays[key - 2])
                else:
                    outputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                        self.paths.basepath,
                        self.dataframe['train'].loc[idx, f'{key}_{decamelize(value.task)}'],
                        **self.instructions['outputs'][key])

            yield inputs, outputs

    def val_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['val'])):
            for key, value in self.data.inputs.items():
                inputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                    self.paths.basepath, self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'],
                    **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                        self.paths.basepath, self.dataframe['val'].loc[idx, f'2_{decamelize(value.task)}'],
                        **self.instructions['outputs'][2])
                    outputs[str(key)] = np.array(arrays[key - 2])
                else:
                    outputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                        self.paths.basepath,
                        self.dataframe['val'].loc[idx, f'{key}_{decamelize(value.task)}'],
                        **self.instructions['outputs'][key])

            yield inputs, outputs

    def test_generator(self):

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['test'])):
            for key, value in self.data.inputs.items():
                inputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                    self.paths.basepath, self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'],
                    **self.instructions['inputs'][key])
            for key, value in self.data.outputs.items():
                if self.data.tags[1].alias == 'object_detection':
                    arrays = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                        self.paths.basepath, self.dataframe['test'].loc[idx, f'2_{decamelize(value.task)}'],
                        **self.instructions['outputs'][2])
                    outputs[str(key)] = np.array(arrays[key - 2])
                else:
                    outputs[str(key)] = getattr(CreateArray(), f"create_{decamelize(value.task)}")(
                        self.paths.basepath,
                        self.dataframe['test'].loc[idx, f'{key}_{decamelize(value.task)}'],
                        **self.instructions['outputs'][key])

            yield inputs, outputs

    def keras_datasets(self):

        (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()

        if self.data.alias in ['mnist', 'fashion_mnist']:
            x_train = x_train[..., None]
            x_val = x_val[..., None]

        if self.data.outputs[2].task == LayerOutputTypeChoice.Classification:
            y_train = utils.to_categorical(y_train, len(np.unique(y_train, axis=0)))
            y_val = utils.to_categorical(y_val, len(np.unique(y_val, axis=0)))

        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=True)
        self.X['train']['1'] = x_train
        self.X['val']['1'] = x_val
        self.X['test']['1'] = x_test
        self.Y['train']['2'] = y_train
        self.Y['val']['2'] = y_val
        self.Y['test']['2'] = y_test

        self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
        self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
        self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        pass

    def prepare_dataset(self):

        def load_arrays():

            for sample in os.listdir(os.path.join(self.paths.arrays)):
                for index in self.data.inputs.keys():
                    put_name = f"{index}"
                    self.X[sample][put_name] = joblib.load(os.path.join(self.paths.arrays, sample, f'{index}.gz'))
                for index, data in self.data.outputs.items():
                    put_name = f"{index}"
                    if data.task == 'ObjectDetection':
                        for i in range(6):
                            self.Y[sample][put_name] = joblib.load(
                                os.path.join(self.paths.arrays, sample, f'{index}.gz'))
                    else:
                        self.Y[sample][put_name] = joblib.load(os.path.join(self.paths.arrays, sample, f'{index}.gz'))

            pass

        if self.data.group == DatasetGroupChoice.keras:
            self.keras_datasets()

        elif self.data.group == DatasetGroupChoice.custom:
            for put in ['train', 'val', 'test']:
                self.dataframe[put] = pd.read_csv(os.path.join(self.paths.instructions, 'tables', f'{put}.csv'),
                                                  index_col=0)

            self.preprocessing.load_preprocesses(list(self.data.inputs.keys()) + list(self.data.outputs.keys()))

            if self.data.use_generator:
                for instr in os.listdir(os.path.join(self.paths.instructions, 'parameters')):
                    with open(os.path.join(self.paths.instructions, 'parameters', instr), 'r') as instruction:
                        ins = json.load(instruction)
                    instr = instr[:instr.rfind('.')]
                    idx, put = instr.split('_')
                    self.instructions[put][int(idx)] = ins

                num_inputs = len(self.data.inputs)
                num_outputs = len(self.data.outputs)
                self.dataset['train'] = Dataset.from_generator(self.train_generator,
                                                               output_shapes=({str(x): self.data.inputs[x].shape for x
                                                                               in range(1, num_inputs + 1)},
                                                                              {str(x): self.data.outputs[x].shape for x
                                                                               in range(num_inputs + 1,
                                                                                        num_outputs + 2)}),
                                                               output_types=({str(x): self.data.inputs[x].dtype for x in
                                                                              range(1, num_inputs + 1)},
                                                                             {str(x): self.data.outputs[x].dtype for x
                                                                              in
                                                                              range(num_inputs + 1, num_outputs + 2)})
                                                               )
                self.dataset['val'] = Dataset.from_generator(self.val_generator,
                                                             output_shapes=({str(x): self.data.inputs[x].shape for x in
                                                                             range(1, num_inputs + 1)},
                                                                            {str(x): self.data.outputs[x].shape for x in
                                                                             range(num_inputs + 1, num_outputs + 2)}),
                                                             output_types=({str(x): self.data.inputs[x].dtype for x in
                                                                            range(1, num_inputs + 1)},
                                                                           {str(x): self.data.outputs[x].dtype for x in
                                                                            range(num_inputs + 1, num_outputs + 2)})
                                                             )
                self.dataset['test'] = Dataset.from_generator(self.test_generator,
                                                              output_shapes=({str(x): self.data.inputs[x].shape for x in
                                                                              range(1, num_inputs + 1)},
                                                                             {str(x): self.data.outputs[x].shape for x
                                                                              in
                                                                              range(num_inputs + 1, num_outputs + 2)}),
                                                              output_types=({str(x): self.data.inputs[x].dtype for x in
                                                                             range(1, num_inputs + 1)},
                                                                            {str(x): self.data.outputs[x].dtype for x in
                                                                             range(num_inputs + 1, num_outputs + 2)})
                                                              )
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        self.dts_prepared = True

        pass
