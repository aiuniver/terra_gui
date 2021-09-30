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
from terra_ai.data.datasets.dataset import DatasetData, DatasetPathsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice, DatasetGroupChoice
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.data.presets.datasets import KerasInstructions


class PrepareDataset(object):

    def __init__(self, data: DatasetData, datasets_path=None):

        self.data = data
        self.language = None
        self.dts_prepared: bool = False
        self.dataframe: dict = {}
        self.instructions: dict = {}
        if self.data.group != 'keras':
            self.paths = DatasetPathsData(basepath=datasets_path)
            self.preprocessing = CreatePreprocessing(dataset_path=self.paths.basepath)

            for put_id in list(self.data.inputs.keys()) + list(self.data.outputs.keys()):
                self.instructions[put_id] = {}
                for instr_json in os.listdir(os.path.join(self.paths.instructions, 'parameters')):
                    idx, *name = os.path.splitext(instr_json)[0].split('_')
                    name = '_'.join(name)
                    if put_id == int(idx):
                        with open(os.path.join(self.paths.instructions, 'parameters', instr_json),
                                  'r') as instr:
                            self.instructions[put_id].update([(f'{idx}_{name}', json.load(instr))])
        else:
            self.preprocessing = CreatePreprocessing()

        self.X: dict = {'train': {}, 'val': {}, 'test': {}}
        self.Y: dict = {'train': {}, 'val': {}, 'test': {}}
        self.dataset: dict = {}

        pass

    def train_generator(self):

        path_type_list = [decamelize(LayerInputTypeChoice.Image), decamelize(LayerOutputTypeChoice.Image),
                          decamelize(LayerInputTypeChoice.Audio), decamelize(LayerOutputTypeChoice.Audio),
                          decamelize(LayerInputTypeChoice.Video), decamelize(LayerOutputTypeChoice.ObjectDetection),
                          decamelize(LayerOutputTypeChoice.Segmentation)]

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['train'])):
            for inp_id in self.data.inputs.keys():
                tmp = []
                for col_name, data in self.instructions[inp_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['train'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['train'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[inp_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                inputs[str(inp_id)] = np.concatenate(tmp, axis=0)

            for out_id in self.data.outputs.keys():
                tmp = []
                for col_name, data in self.instructions[out_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['train'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['train'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[out_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                outputs[str(out_id)] = np.concatenate(tmp, axis=0)

            yield inputs, outputs

    def val_generator(self):

        path_type_list = [decamelize(LayerInputTypeChoice.Image), decamelize(LayerOutputTypeChoice.Image),
                          decamelize(LayerInputTypeChoice.Audio), decamelize(LayerOutputTypeChoice.Audio),
                          decamelize(LayerInputTypeChoice.Video), decamelize(LayerOutputTypeChoice.ObjectDetection),
                          decamelize(LayerOutputTypeChoice.Segmentation)]

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['val'])):
            for inp_id in self.data.inputs.keys():
                tmp = []
                for col_name, data in self.instructions[inp_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['val'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['val'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[inp_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                inputs[str(inp_id)] = np.concatenate(tmp, axis=0)

            for out_id in self.data.outputs.keys():
                tmp = []
                for col_name, data in self.instructions[out_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['val'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['val'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[out_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                outputs[str(out_id)] = np.concatenate(tmp, axis=0)

            yield inputs, outputs

    def test_generator(self):

        path_type_list = [decamelize(LayerInputTypeChoice.Image), decamelize(LayerOutputTypeChoice.Image),
                          decamelize(LayerInputTypeChoice.Audio), decamelize(LayerOutputTypeChoice.Audio),
                          decamelize(LayerInputTypeChoice.Video), decamelize(LayerOutputTypeChoice.ObjectDetection),
                          decamelize(LayerOutputTypeChoice.Segmentation)]

        inputs = {}
        outputs = {}
        for idx in range(len(self.dataframe['test'])):
            for inp_id in self.data.inputs.keys():
                tmp = []
                for col_name, data in self.instructions[inp_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['test'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['test'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[inp_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                inputs[str(inp_id)] = np.concatenate(tmp, axis=0)

            for out_id in self.data.outputs.keys():
                tmp = []
                for col_name, data in self.instructions[out_id].items():
                    if data['put_type'] in path_type_list:
                        sample = os.path.join(self.paths.basepath, self.dataframe['test'].loc[idx, col_name])
                    else:
                        sample = self.dataframe['test'].loc[idx, col_name]
                    array = getattr(CreateArray(), f'create_{data["put_type"]}')(sample, **{
                        'preprocess': self.preprocessing.preprocessing[out_id][col_name]}, **data)
                    array = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(array['instructions'],
                                                                                     **array['parameters'])
                    tmp.append(array)
                outputs[str(out_id)] = np.concatenate(tmp, axis=0)

            yield inputs, outputs

    def keras_datasets(self):

        (x_train, y_train), (x_val, y_val) = getattr(load_keras_datasets, self.data.alias).load_data()

        if self.data.alias in ['mnist', 'fashion_mnist']:
            x_train = x_train[..., None]
            x_val = x_val[..., None]
        for out in self.data.outputs.keys():
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

            for put_id, data in KerasInstructions[self.data.alias].items():
                self.instructions[put_id] = data

            if self.data.alias in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
                self.preprocessing.create_scaler(**{'put': 1, 'scaler': 'min_max_scaler',
                                                    'min_scaler': 0, 'max_scaler': 1,
                                                    'cols_names': f'1_{self.data.alias}'})
                self.preprocessing.preprocessing[1][f'1_{self.data.alias}'].fit(self.X['train']['1'].reshape(-1, 1))
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

                num_inputs = len(self.data.inputs)
                num_outputs = len(self.data.outputs)
                self.dataset['train'] = Dataset.from_generator(self.train_generator,
                                                               output_shapes=({str(x): self.data.inputs[x].shape for x
                                                                               in range(1, num_inputs+1)},
                                                                              {str(x): self.data.outputs[x].shape for x
                                                                               in range(num_inputs + 1,
                                                                                        num_outputs + 2)}),
                                                               output_types=({str(x): self.data.inputs[x].dtype for x
                                                                              in range(1, num_inputs + 1)},
                                                                             {str(x): self.data.outputs[x].dtype for x
                                                                              in range(num_inputs + 1,
                                                                                       num_outputs + 2)})
                                                               )
                self.dataset['val'] = Dataset.from_generator(self.val_generator,
                                                             output_shapes=({str(x): self.data.inputs[x].shape for x
                                                                             in range(1, num_inputs + 1)},
                                                                            {str(x): self.data.outputs[x].shape for x
                                                                             in range(num_inputs + 1,
                                                                                      num_outputs + 2)}),
                                                             output_types=({str(x): self.data.inputs[x].dtype for x
                                                                            in range(1, num_inputs + 1)},
                                                                           {str(x): self.data.outputs[x].dtype for x
                                                                            in range(num_inputs + 1,
                                                                                     num_outputs + 2)})
                                                             )
                self.dataset['test'] = Dataset.from_generator(self.test_generator,
                                                              output_shapes=({str(x): self.data.inputs[x].shape for x
                                                                              in range(1, num_inputs + 1)},
                                                                             {str(x): self.data.outputs[x].shape for x
                                                                              in range(num_inputs + 1,
                                                                                       num_outputs + 2)}),
                                                              output_types=({str(x): self.data.inputs[x].dtype for x
                                                                             in range(1, num_inputs + 1)},
                                                                            {str(x): self.data.outputs[x].dtype for x
                                                                             in range(num_inputs + 1,
                                                                                      num_outputs + 2)})
                                                              )
            else:
                load_arrays()

                self.dataset['train'] = Dataset.from_tensor_slices((self.X['train'], self.Y['train']))
                self.dataset['val'] = Dataset.from_tensor_slices((self.X['val'], self.Y['val']))
                self.dataset['test'] = Dataset.from_tensor_slices((self.X['test'], self.Y['test']))

        self.dts_prepared = True

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

        with open(os.path.join(folder_path, 'config.json'), 'w') as cfg:
            json.dump(self.data.native(), cfg)
