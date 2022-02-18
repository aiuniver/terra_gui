import os
import numpy as np
import joblib
import json
import pandas as pd
import tensorflow as tf
import h5py

from tensorflow.keras import utils
from tensorflow.keras import datasets as load_keras_datasets
from tensorflow.data import Dataset
# from PIL import Image

# from terra_ai.data.training.extra import ArchitectureChoice
# from terra_ai.utils import decamelize
from terra_ai.datasets.preprocessing import CreatePreprocessing
# from terra_ai.datasets import arrays_create
# from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.data.datasets.dataset import DatasetData, DatasetPathsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice
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

        self.X: dict = {'train': {}, 'val': {}}
        self.Y: dict = {'train': {}, 'val': {}}
        self.service: dict = {'train': {}, 'val': {}}

        self.dataset: dict = {}

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
                        self.X[split][str(index)] = joblib.load(self.paths.arrays.joinpath(split, f'{index}.gz'))
                    for index in self.data.outputs.keys():
                        self.Y[split][str(index)] = joblib.load(self.paths.arrays.joinpath(split, f'{index}.gz'))
                    if self.data.service:
                        for index in self.data.service.keys():
                            self.service[split][str(index)] = joblib.load(
                                self.paths.arrays.joinpath(split, f'{index}_service.gz')
                            )

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

        dataset_data = self.data.native()
        dataset_data.update({"instructions": self.instructions})

        with open(os.path.join(folder_path, 'dataset.json'), 'w') as cfg:
            json.dump(dataset_data, cfg)
