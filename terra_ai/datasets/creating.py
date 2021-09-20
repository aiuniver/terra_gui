from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice, \
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

        self.columns_processing = {}
        if creation_data.columns_processing:
            for key, value in creation_data.columns_processing.items():
                self.columns_processing[key] = value

        self.instructions: DatasetInstructionsData = self.create_instructions(creation_data)
        self.create_preprocessing(self.instructions)
        self.create_table(creation_data=creation_data)

        self.inputs: dict = self.create_input_parameters(creation_data=creation_data)
        self.outputs: dict = self.create_output_parameters(creation_data=creation_data)

        if not creation_data.use_generator:
            self.x_array = self.create_dataset_arrays(put_data=self.instructions.inputs)
            self.y_array = self.create_dataset_arrays(put_data=self.instructions.outputs)
            self.write_arrays(self.x_array, self.y_array)

        # self.write_preprocesses_to_files()
        self.write_instructions_to_files(creation_data=creation_data)
        self.datasetdata = DatasetData(**self.write_dataset_configure(creation_data=creation_data))

        shutil.rmtree(self.temp_directory)
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
                else:
                    for inp in creation_data.inputs:
                        if inp.type == LayerInputTypeChoice.Dataframe:
                            inp.parameters.y_cols = out.parameters.cols_names
                            out.parameters.xlen_step = inp.parameters.xlen_step
                            out.parameters.xlen = inp.parameters.xlen
                            out.parameters.step_len = inp.parameters.step_len
                            out.parameters.separator = inp.parameters.separator
            elif out.type == LayerOutputTypeChoice.Regression:
                for inp in creation_data.inputs:
                    if inp.type == LayerInputTypeChoice.Dataframe:
                        inp.parameters.y_cols = out.parameters.cols_names
                        # out.parameters.xlen_step = inp.parameters.xlen_step
                        # out.parameters.xlen = inp.parameters.xlen
                        # out.parameters.step_len = inp.parameters.step_len
                        # out.parameters.separator = inp.parameters.separator
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
            elif out.type == LayerOutputTypeChoice.Timeseries:
                for inp in creation_data.inputs:
                    if inp.type == LayerInputTypeChoice.Dataframe:
                        out.parameters.transpose = inp.parameters.transpose
                        out.parameters.separator = inp.parameters.separator
                        inp.parameters.step = out.parameters.step
                        inp.parameters.length = out.parameters.length
                        inp.parameters.trend = out.parameters.trend
                        inp.parameters.depth = out.parameters.depth
            elif out.type == LayerOutputTypeChoice.ObjectDetection:
                with open(creation_data.source_path.joinpath('obj.names'), 'r') as names:
                    names_list = names.read()
                names_list = [elem for elem in names_list.split('\n') if elem]
                out.parameters.classes_names = names_list
                out.parameters.num_classes = len(names_list)

        if creation_data.columns_processing:
            for worker_name, worker_params in creation_data.columns_processing.items():
                if creation_data.columns_processing[worker_name].type == 'ImageSegmentation':
                    for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type == 'Image':
                            creation_data.columns_processing[worker_name].parameters.height = \
                                creation_data.columns_processing[w_name].parameters.height
                            creation_data.columns_processing[worker_name].parameters.width = \
                                creation_data.columns_processing[w_name].parameters.width

        return creation_data

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:

        if self.columns_processing:
            inputs = self.create_dataframe_put_instructions(data=creation_data.inputs)
            outputs = self.create_dataframe_put_instructions(data=creation_data.outputs)
        else:
            inputs = self.create_put_instructions(data=creation_data.inputs)
            outputs = self.create_put_instructions(data=creation_data.outputs)
            for out in creation_data.outputs:
                if out.type == LayerOutputTypeChoice.Classification and not out.parameters.cols_names:
                    outputs[out.id].instructions = self.y_cls
        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_dataframe_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]):

        put_parameters = {}

        for put in data:
            self.tags[put.id] = {}
            put_columns = {}
            cols_names = list(put.parameters.cols_names.keys())
            dataframe = pd.read_csv(put.parameters.sources_paths[0], usecols=cols_names)
            for name in cols_names:
                instructions_data = None
                for worker in put.parameters.cols_names[name]:  # На будущее после 1 октября - очень аккуратно!
                    self.tags[put.id][f'{put.id}_{name}'] = decamelize(self.columns_processing[worker].type)
                    list_of_data = dataframe.loc[:, name].to_numpy().tolist()
                    instr = getattr(CreateArray(),
                                    f'instructions_{decamelize(self.columns_processing[worker].type)}')(
                        list_of_data, **{'cols_names': f'{put.id}_{name}', 'id': put.id},
                        **self.columns_processing[worker].parameters.native())
                    paths_list = [os.path.join(self.source_path, elem) for elem in instr['instructions']]
                    instructions_data = InstructionsData(
                        **getattr(CreateArray(),
                                  f"cut_{decamelize(self.columns_processing[worker].type)}")(paths_list, self.temp_directory, os.path.join(self.paths.sources, f'{put.id}_{name}'), **instr['parameters']))
                    instructions_data.instructions = [
                        os.path.join('sources', instructions_data.parameters['cols_names'],
                                     path.replace(str(self.source_path) + os.path.sep, '')) for path in
                        instructions_data.instructions]
                put_columns[f'{put.id}_{name}'] = instructions_data
            put_parameters[put.id] = put_columns

        return put_parameters

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

            temp_paths_list = [os.path.join(self.source_path, x) for x in paths_list]
            instr = getattr(CreateArray(), f"instructions_{decamelize(elem.type)}")(temp_paths_list, **elem.native())

            if (not elem.type == LayerOutputTypeChoice.Classification) and ('dataframe' not in self.tags.values()):
                y_classes = sorted(list(instr['instructions'].keys())) if \
                    isinstance(instr['instructions'], dict) else instr['instructions']
                self.y_cls = [os.path.basename(os.path.dirname(dir_name)) for dir_name in y_classes]

            instructions_data = InstructionsData(
                **getattr(CreateArray(), f"cut_{decamelize(elem.type)}")(instr['instructions'], self.temp_directory,
                                                                         os.path.join(self.paths.sources,
                                                                                      f"{elem.id}_{decamelize(elem.type)}"),
                                                                         **instr['parameters']))
            if 'dataframe' not in self.tags.values():
                if elem.type not in [LayerInputTypeChoice.Text, LayerOutputTypeChoice.Text,
                                     LayerOutputTypeChoice.TextSegmentation, LayerOutputTypeChoice.Regression]:
                    if elem.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Segmentation,
                                     LayerOutputTypeChoice.Image, LayerOutputTypeChoice.ObjectDetection]:
                        new_paths = [os.path.join('sources', f'{elem.id}_{decamelize(elem.type)}',
                                                  path.replace(self.source_directory + os.path.sep, '')) for path in
                                     instructions_data.instructions]
                    else:
                        new_paths = [os.path.join('sources', path.replace(self.temp_directory + os.path.sep, '')) for
                                     path in instructions_data.instructions]
                    instructions_data.instructions = new_paths

            instructions.update([(elem.id, instructions_data)])

        return instructions

    def create_preprocessing(self, instructions: DatasetInstructionsData):

        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            for col_name, data in put.items():
                if 'scaler' in data.parameters.keys():
                    if data.parameters['scaler'] != LayerScalerImageChoice.no_scaler:
                        self.preprocessing.create_scaler(array=None, **data.parameters)
                elif 'prepare_method' in data.parameters.keys():
                    if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                             LayerPrepareMethodChoice.bag_of_words]:
                        self.preprocessing.create_tokenizer(array=None, **data.parameters)
                    elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                        self.preprocessing.create_word2vec(array=None, **data.parameters)


    def create_table(self, creation_data: CreationData):

        for elem in creation_data.outputs:
            classes = self.instructions.outputs[elem.id][list(
                self.instructions.outputs[elem.id].keys())[0]].instructions
            if elem.type == LayerOutputTypeChoice.Classification:
                peg = [0]
                prev_cls = classes[0]
                for idx, x in enumerate(classes):
                    if x != prev_cls:
                        peg.append(idx)
                        prev_cls = x
                peg.append(len(classes))
            else:
                peg = [0, len(classes)]

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
        for inp in self.instructions.inputs.keys():
            for key, value in self.instructions.inputs[inp].items():
                build_dataframe[key] = value.instructions
        for out in self.instructions.outputs.keys():
            for key, value in self.instructions.outputs[out].items():
                build_dataframe[key] = value.instructions

        dataframe = pd.DataFrame(build_dataframe)
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)

    def create_input_parameters(self, creation_data: CreationData) -> dict:

        creating_inputs_data = {}
        for key in self.instructions.inputs.keys():
            classes_names = [os.path.basename(x) for x in creation_data.inputs.get(key).parameters.sources_paths]
            num_classes = len(classes_names)
            # if creation_data.inputs.get(key).type == LayerInputTypeChoice.Text:
            #     arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
            #         self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
            #         **self.instructions.inputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            #     array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])
            # else:
            full_array = []
            for col_name, data in self.instructions.inputs[key].items():
                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                    os.path.join(self.paths.basepath, data.instructions[0]), **data.parameters) #  , **self.preprocessing.preprocessing.get(key))
                full_array.append(getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                         **arr['parameters']))

            array = np.concatenate(full_array, axis=0)

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
            # if (creation_data.outputs.get(key).type in
            #     [LayerOutputTypeChoice.Text, LayerOutputTypeChoice.TextSegmentation]) or (
            #         'dataframe' in self.tags.values()):
            #     arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
            #         self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
            #         **self.instructions.outputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
            #     array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])
            #     if 'classification' in self.tags.values():
            #         cl_names = self.instructions.outputs.get(key).parameters['classes_names']
            #         classes_names = cl_names if cl_names else [os.path.basename(x) for x in creation_data.outputs.get(
            #             key).parameters.sources_paths]
            #         num_classes = len(classes_names)
            #     else:
            #         classes_names = None
            #         num_classes = None
            #
            # else:
            classes_names, classes_colors, num_classes, encoding = None, None, None, None
            full_array = []
            for col_name, data in self.instructions.outputs[key].items():
                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                    os.path.join(self.paths.basepath, data.instructions[0]),
                    **data.parameters)  # , **self.preprocessing.preprocessing.get(key))
                full_array.append(getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                                   **arr['parameters']))
                cl_names = data.parameters.get('classes_names')
                classes_names = cl_names if cl_names else [os.path.basename(x) for x in
                                                           creation_data.outputs.get(key).parameters.sources_paths]
                num_classes = len(classes_names)

                classes_colors = data.parameters.get('classes_colors')
                if data.parameters.get('encoding'):
                    encoding = data.parameters.get('encoding')
                elif data.parameters.get('one_hot_encoding'):
                    if data.parameters.get('one_hot_encoding'):
                        encoding = LayerEncodingChoice.ohe
                    else:
                        encoding = LayerEncodingChoice.none

                elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.Segmentation:
                    encoding = LayerEncodingChoice.ohe
                elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.TextSegmentation:
                    encoding = LayerEncodingChoice.multi
                else:
                    encoding = LayerEncodingChoice.none

            array = np.concatenate(full_array, axis=0)

            iters = 1 if isinstance(array, np.ndarray) else len(array)
            array = np.expand_dims(array, 0) if isinstance(array, np.ndarray) else array
            for i in range(iters):
                current_output = DatasetOutputsData(datatype=DataType.get(len(array[i].shape), 'DIM'),
                                                    dtype=str(array[i].dtype),
                                                    shape=array[i].shape,
                                                    name=creation_data.outputs.get(key).name,
                                                    task=creation_data.outputs.get(key).type,
                                                    classes_names=classes_names,
                                                    classes_colors=classes_colors,
                                                    num_classes=num_classes,
                                                    encoding=encoding
                                                    )
                creating_outputs_data.update([(key + i, current_output.native())])

        return creating_outputs_data

    def create_dataset_arrays(self, put_data: dict) -> dict:

        out_array = {'train': {}, 'val': {}, 'test': {}}
        for split in list(out_array.keys()):
            for key in put_data.keys():
                current_arrays: list = []
                # if self.tags[key] == 'object_detection':
                #     num_arrays = 6
                #     for i in range(num_arrays):
                #         globals()[f'current_arrays_{i + 1}'] = []
                #     for i in range(len(self.dataframe[split])):
                #         arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                #             os.path.join(self.paths.basepath,
                #                             self.dataframe[split].loc[i, f'{key}_{self.tags[key]}']),
                #             **put_data.get(key).parameters,
                #             **self.preprocessing.preprocessing.get(key))
                #         array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                #                                                                         **arr['parameters'])
                #         for j in range(num_arrays):
                #             globals()[f'current_arrays_{j + 1}'].append(array[j])

                # else:
                for i in range(len(self.dataframe[split])):
                    # if self.tags[key] in [decamelize(LayerInputTypeChoice.Text),
                    #                       decamelize(LayerOutputTypeChoice.Text),
                    #                       decamelize(LayerOutputTypeChoice.TextSegmentation)]:
                    #     arr = getattr(CreateArray(), f'create_{self.tags[key]}')(self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'], **put_data.get(key).parameters) #  , **self.preprocessing.preprocessing.get(key))
                    #     array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                    #                                                                     **arr['parameters'])
                    # else:
                    full_array = []
                    for col_name, data in put_data[key].items():
                        arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                            os.path.join(self.paths.basepath, self.dataframe[split].loc[i, col_name]),
                            **data.parameters)  # , **self.preprocessing.preprocessing.get(key))
                        full_array.append(getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(
                            arr['instructions'], **arr['parameters']))

                    array = np.concatenate(full_array, axis=0)

                    current_arrays.append(array)
                # if self.tags[key] == 'object_detection':
                #     for n in range(num_arrays):
                #         out_array[split][key + n] = np.array(globals()[f'current_arrays_{n + 1}'])
                # else:
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
                inp = inp.native()
                del inp['parameters']['sources_paths']
                json.dump(inp, cfg)
        for out in creation_data.outputs:
            with open(os.path.join(parameters_path, f'{out.id}_outputs.json'), 'w') as cfg:
                out = out.native()
                del out['parameters']['sources_paths']
                json.dump(out, cfg)

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
        for put in self.tags.values():
            for value in put.values():
                tags_list.append({'alias': value, 'name': value.title()})
        for tag in creation_data.tags:
            tags_list.append(tag.native())

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
