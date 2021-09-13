from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice,\
    LayerPrepareMethodChoice, LayerScalerImageChoice
from terra_ai.utils import decamelize
from terra_ai.datasets.data import DataType, InstructionsData, DatasetInstructionsData
from terra_ai.data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList
from terra_ai.data.datasets.dataset import DatasetData, DatasetInputsData, DatasetOutputsData, DatasetPathsData
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.data.datasets.extra import LayerEncodingChoice

import os
import random
import numpy as np
import pandas as pd
import json
import joblib
import tempfile
import shutil
from pathlib import Path
from typing import Union
from datetime import datetime
from pytz import timezone


class CreateDataset(object):

    def __init__(self, cr_data: CreationData):

        creation_data = self.preprocess_creation_data(cr_data)

        os.makedirs(Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}'), exist_ok=True)
        self.paths = DatasetPathsData(
            basepath=Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}'))
        self.temp_directory = tempfile.mkdtemp()

        self.source_directory: str = str(creation_data.source_path)
        self.dataframe: dict = {}
        self.temporary: dict = {}
        self.tags: dict = {}
        self.preprocessing = CreatePreprocessing()
        self.use_generator: bool = False
        self.source_path = creation_data.source_path  # исправить
        self.y_cls: list = []

        self.instructions: DatasetInstructionsData = self.create_instructions(creation_data)
        self.create_preprocessing(self.instructions)
        self.create_table(creation_data=creation_data)

        self.inputs: dict = self.create_input_parameters(creation_data=creation_data)
        self.outputs: dict = self.create_output_parameters(creation_data=creation_data)

        if not creation_data.use_generator:
            x_array = self.create_dataset_arrays(put_data=self.instructions.inputs)
            y_array = self.create_dataset_arrays(put_data=self.instructions.outputs)
            self.write_arrays(x_array, y_array)

        self.write_preprocesses_to_files()
        self.write_instructions_to_files(creation_data=creation_data)
        self.datasetdata = DatasetData(**self.write_dataset_configure(creation_data=creation_data))

        shutil.rmtree(self.temp_directory)

        # self.minvalue_y: int = 0
        # self.maxvalue_y: int = 0

        # self.minvalues: dict = {}
        # self.maxvalues: dict = {}

        pass

    @staticmethod
    def preprocess_creation_data(creation_data):

        for out in creation_data.outputs:
            if out.type == LayerOutputTypeChoice.Classification:
                if not out.parameters.sources_paths or not out.parameters.sources_paths[0].suffix == '.csv':
                    for inp in creation_data.inputs:
                        if inp.type in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
                                        LayerInputTypeChoice.Audio, LayerInputTypeChoice.Video]:
                            out.parameters.sources_paths = inp.parameters.sources_paths
                        break
            elif out.type == LayerOutputTypeChoice.Segmentation:
                for inp in creation_data.inputs:
                    if inp.type == LayerInputTypeChoice.Image:
                        out.parameters.width = inp.parameters.width
                        out.parameters.height = inp.parameters.height
            elif out.type == LayerOutputTypeChoice.TextSegmentation:
                for inp in creation_data.inputs:
                    if inp.type == LayerOutputTypeChoice.Text:
                        out.parameters.sources_paths = inp.parameters.sources_paths
                        out.parameters.text_mode = inp.parameters.text_mode
                        out.parameters.length = inp.parameters.length
                        out.parameters.step = inp.parameters.step
                        out.parameters.max_words = inp.parameters.max_words
                        out.parameters.filters = inp.parameters.filters
                        inp.parameters.open_tags = out.parameters.open_tags
                        inp.parameters.close_tags = out.parameters.close_tags

        return creation_data

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:

        inputs = self.create_put_instructions(data=creation_data.inputs)
        outputs = self.create_put_instructions(data=creation_data.outputs)
        for out in creation_data.outputs:
            if out.type == LayerOutputTypeChoice.Classification and not out.parameters.cols_names:
                outputs[out.id].instructions = self.y_cls
        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:

        instructions: dict = {}
        for elem in data:
            self.tags[elem.id] = decamelize(elem.type)
            paths_list: list = []
            for paths in elem.parameters.sources_paths:
                if paths.is_dir():
                    for directory, folder, file_name in sorted(os.walk(os.path.join(self.source_directory, paths))):
                        if file_name:
                            file_folder = directory.replace(self.source_directory, '')[1:]
                            for name in sorted(file_name):
                                paths_list.append(os.path.join(file_folder, name))
                elif paths.is_file() and paths.suffix == '.csv' and elem.type not in [LayerInputTypeChoice.Dataframe,
                                                                                      LayerOutputTypeChoice.Timeseries]:
                    data = pd.read_csv(os.path.join(self.source_directory, paths), usecols=elem.parameters.cols_names)
                    paths_list = data[elem.parameters.cols_names[0]].to_list()
            temp_paths_list = [os.path.join(self.source_path, x) for x in paths_list]
            instr = getattr(CreateArray(), f"instructions_{decamelize(elem.type)}")(temp_paths_list, **elem.native())

            if not elem.type == LayerOutputTypeChoice.Classification:
                y_classes = sorted(list(instr['instructions'].keys())) if\
                    isinstance(instr['instructions'], dict) else instr['instructions']
                self.y_cls = [os.path.basename(os.path.dirname(dir_name)) for dir_name in y_classes]

            instructions_data = InstructionsData(
                **getattr(CreateArray(), f"cut_{decamelize(elem.type)}")(instr['instructions'], self.temp_directory,
                                                                         os.path.join(self.paths.sources,
                                                                                      f"{elem.id}_{decamelize(elem.type)}"),
                                                                         **instr['parameters']))
            if elem.type not in [LayerInputTypeChoice.Text, LayerOutputTypeChoice.Text,
                                 LayerOutputTypeChoice.TextSegmentation, LayerOutputTypeChoice.Regression]:
                if elem.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Image,
                                 LayerOutputTypeChoice.Segmentation]:
                    new_paths = [os.path.join('sources', f'{elem.id}_{decamelize(elem.type)}',
                                              path.replace(self.source_directory + os.path.sep, '')) for path in
                                 instructions_data.instructions]
                else:
                    new_paths = [os.path.join('sources', path.replace(self.temp_directory + os.path.sep, '')) for path
                                 in instructions_data.instructions]
                instructions_data.instructions = new_paths

            instructions.update([(elem.id, instructions_data)])

        return instructions

    def create_preprocessing(self, instructions: DatasetInstructionsData):

        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            if 'scaler' in put.parameters.keys() and put.parameters['scaler'] != LayerScalerImageChoice.no_scaler:
                self.preprocessing.create_scaler(put.parameters['put'], **put.parameters)
            elif 'prepare_method' in put.parameters.keys():
                if put.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                        LayerPrepareMethodChoice.bag_of_words]:
                    self.preprocessing.create_tokenizer(put.parameters['put'], put.instructions, **put.parameters)
                elif put.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                    self.preprocessing.create_word2vec(put.parameters['put'], put.instructions, **put.parameters)
            else:
                self.preprocessing.create_dull(put.parameters['put'])

    def create_table(self, creation_data: CreationData):

        peg = [0]
        for elem in creation_data.outputs:
            classes = self.instructions.outputs.get(elem.id).instructions
            if elem.type == LayerOutputTypeChoice.Classification:
                prev_cls = classes[0]
                for idx, x in enumerate(classes):
                    if x != prev_cls:
                        peg.append(idx)
                        prev_cls = x
                peg.append(len(classes))
            else:
                peg.append(len(classes))

        split_sequence = {"train": [], "val": [], "test": []}
        for i in range(len(peg) - 1):
            indices = np.arange(peg[i], peg[i + 1])
            train_len = int(creation_data.info.part.train * len(indices))
            val_len = int(creation_data.info.part.validation * len(indices))
            indices = indices.tolist()
            split_sequence["train"].extend(indices[:train_len])
            split_sequence["val"].extend(indices[train_len: train_len + val_len])
            split_sequence["test"].extend(indices[train_len + val_len:])
        if creation_data.info.shuffle:
            random.shuffle(split_sequence["train"])
            random.shuffle(split_sequence["val"])
            random.shuffle(split_sequence["test"])

        build_dataframe = {}
        for key in self.instructions.inputs.keys():
            build_dataframe[f'{key}_{self.tags[key]}'] = self.instructions.inputs[key].instructions
        for key in self.instructions.outputs.keys():
            build_dataframe[f'{key}_{self.tags[key]}'] = self.instructions.outputs[key].instructions
        dataframe = pd.DataFrame(build_dataframe)
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)

    def create_input_parameters(self, creation_data: CreationData) -> dict:

        creating_inputs_data = {}
        for key in self.instructions.inputs.keys():
            # if self.tags[key] == "dataframe":
            #     array = getattr(CreateArray(), f"create_{self.tags[key]}")(
            #         # creation_data.source_path,
            #         self.instructions.inputs.get(key).instructions[0],
            #         **self.instructions.inputs.get(key).parameters,
            #     )
            # else:
            classes_names = [os.path.basename(x) for x in creation_data.inputs.get(key).parameters.sources_paths]
            num_classes = len(classes_names)
            if creation_data.inputs.get(key).type == LayerInputTypeChoice.Text:
                array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
                    **self.instructions.inputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            else:
                array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    os.path.join(self.paths.basepath, self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}']),
                    **self.instructions.inputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            current_input = DatasetInputsData(datatype=DataType.get(len(array.shape), 'DIM'),
                                              dtype=str(array.dtype),
                                              shape=array.shape,
                                              name=creation_data.inputs.get(key).name,
                                              task=creation_data.inputs.get(key).type,
                                              classes_names=classes_names,
                                              num_classes=num_classes,
                                              encoding=LayerEncodingChoice.none
                                              )
            creating_inputs_data.update([(key, current_input.native())])

        return creating_inputs_data

    def create_output_parameters(self, creation_data: CreationData) -> dict:
        creating_outputs_data = {}
        for key in self.instructions.outputs.keys():
            # if self.tags[key] == "timeseries":
            #     array = getattr(CreateArray(), f"create_{self.tags[key]}")(
            #         creation_data.source_path,
            #         self.instructions.outputs.get(key).instructions[0],
            #         **self.instructions.outputs.get(key).parameters,
            #     )
            # else:
            if creation_data.outputs.get(key).type in [LayerOutputTypeChoice.Text,
                                                       LayerOutputTypeChoice.TextSegmentation]:
                array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
                    **self.instructions.outputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            else:
                array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    os.path.join(self.paths.basepath, self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}']),
                    **self.instructions.outputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            cl_names = self.instructions.outputs.get(key).parameters[
                'classes_names']  # creation_data.outputs.get(key).parameters.__dict__.get('classes_names')
            classes_names = cl_names if cl_names else [os.path.basename(x) for x in creation_data.outputs.get(
                key).parameters.sources_paths]
            num_classes = len(classes_names)
            if self.instructions.outputs.get(key).parameters.get('encoding'):
                encoding = self.instructions.outputs.get(key).parameters.get('encoding')
            elif self.instructions.outputs.get(key).parameters.get('one_hot_encoding'):
                if self.instructions.outputs.get(key).parameters.get('one_hot_encoding'):
                    encoding = LayerEncodingChoice.ohe
                else:
                    encoding = LayerEncodingChoice.none
            elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.Segmentation:
                encoding = LayerEncodingChoice.ohe
            elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.TextSegmentation:
                encoding = LayerEncodingChoice.multi
            else:
                encoding = LayerEncodingChoice.none
            iters = 1 if isinstance(array, np.ndarray) else len(array)
            array = np.expand_dims(array, 0) if isinstance(array, np.ndarray) else array
            for i in range(iters):
                current_output = DatasetOutputsData(datatype=DataType.get(len(array[i].shape), 'DIM'),
                                                    dtype=str(array[i].dtype),
                                                    shape=array[i].shape,
                                                    name=creation_data.outputs.get(key).name,
                                                    task=creation_data.outputs.get(key).type,
                                                    classes_names=classes_names,
                                                    num_classes=num_classes,
                                                    encoding=encoding
                                                    )
                creating_outputs_data.update([(key + i, current_output.native())])

        return creating_outputs_data

    def create_dataset_arrays(self, put_data: dict) -> dict:

        out_array = {'train': {}, 'val': {}, 'test': {}}
        # num_arrays = 1
        for split in list(out_array.keys()):
            for key in put_data.keys():
                current_arrays: list = []
                if self.tags[key] == 'object_detection':
                    num_arrays = 6
                    for i in range(num_arrays):
                        globals()[f'current_arrays_{i + 1}'] = []

                for i in range(len(self.dataframe[split])):
                    # if self.tags[key] in ['dataframe', 'timeseries']:
                    #     array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    #         os.path.join(self.paths.basepath, put_data.get(key).instructions[i]),
                    #         **put_data.get(key).parameters
                    #     )
                    # else:
                    if self.tags[key] in [decamelize(LayerInputTypeChoice.Text), decamelize(LayerOutputTypeChoice.Text),
                                          decamelize(LayerOutputTypeChoice.TextSegmentation)]:
                        array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                            self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'], **put_data.get(key).parameters,
                            **self.preprocessing.preprocessing.get(key))
                    else:
                        array = getattr(CreateArray(), f'create_{self.tags[key]}')(
                            os.path.join(self.paths.basepath, self.dataframe[split].loc[i, f'{key}_{self.tags[key]}']),
                            **put_data.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                    # if self.tags[key] == 'object_detection':
                    #     for j in range(num_arrays):
                    #         globals()[f'current_arrays_{j + 1}'].append(array[j])
                    # else:
                    current_arrays.append(array)

                # if self.tags[key] == 'object_detection':
                #     for j in range(num_arrays):
                #         out_array[split][key + j] = np.array(globals()[f'current_arrays_{j + 1}'])
                # else:
                print(np.array(current_arrays).shape)
                out_array[split][key] = np.array(current_arrays)

        return out_array

    def write_arrays(self, array_x, array_y):

        for array in [array_x, array_y]:
            for sample in array.keys():
                for inp in array[sample].keys():
                    os.makedirs(os.path.join(self.paths.arrays, sample), exist_ok=True)
                    joblib.dump(array[sample][inp], os.path.join(self.paths.arrays, sample, f'{inp}.gz'))

    def write_instructions_to_files(self, creation_data: CreationData):

        parameters_path = os.path.join(self.paths.instructions, 'parameters')
        tables_path = os.path.join(self.paths.instructions, 'tables')

        os.makedirs(parameters_path, exist_ok=True)
        for inp in creation_data.inputs:
            with open(os.path.join(parameters_path, f'{inp.id}_inputs.json'), 'w') as cfg:
                json.dump(inp.native(), cfg)
        for out in creation_data.outputs:
            with open(os.path.join(parameters_path, f'{out.id}_outputs.json'), 'w') as cfg:
                json.dump(out.native(), cfg)

        os.makedirs(tables_path, exist_ok=True)
        for key in self.dataframe.keys():
            self.dataframe[key].to_csv(os.path.join(self.paths.instructions, 'tables', f'{key}.csv'))

        pass

    def write_preprocesses_to_files(self):

        for put in self.preprocessing.preprocessing.keys():
            for param in self.preprocessing.preprocessing[put]:
                if self.preprocessing.preprocessing[put][param] and param != 'dull':
                    os.makedirs(self.paths.__dict__[param.split('_')[1]], exist_ok=True)
                    joblib.dump(self.preprocessing.preprocessing[put][param],
                                os.path.join(self.paths.__dict__[param.split('_')[1]], f'{put}.gz'))

        pass

    def write_dataset_configure(self, creation_data: CreationData) -> dict:

        # Размер датасета
        size_bytes = 0
        for path, dirs, files in os.walk(self.paths.basepath):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        # Теги в нужном формате
        tags_list = []
        for value in self.tags.values():
            tags_list.append({'alias': value, 'name': value.title()})

        data = {'name': creation_data.name,
                'alias': creation_data.alias,
                'group': DatasetGroupChoice.custom,
                'use_generator': creation_data.use_generator,
                'tags': tags_list,
                'user_tags': creation_data.tags,
                'language': '',  # зачем?...
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'size': {'value': size_bytes}
                }

        for attr in ["inputs", "outputs"]:
            data[attr] = self.__dict__[attr]

        with open(os.path.join(self.paths.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(DatasetData(**data).native(), fp)
        print(DatasetData(**data).native())

        return data
