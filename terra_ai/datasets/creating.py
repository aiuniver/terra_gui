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

        self.instructions: DatasetInstructionsData = self.create_instructions(creation_data)
        self.create_preprocessing(self.instructions)
        self.create_table(creation_data=creation_data)

        self.inputs: dict = self.create_input_parameters(creation_data=creation_data)
        self.outputs: dict = self.create_output_parameters(creation_data=creation_data)

        if not creation_data.use_generator:
            self.x_array = self.create_dataset_arrays(put_data=self.instructions.inputs)  # TODO
            self.y_array = self.create_dataset_arrays(put_data=self.instructions.outputs) # TODO
            self.write_arrays(self.x_array, self.y_array)

        self.write_preprocesses_to_files()
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
                    paths_list = data.iloc[:, 0].tolist()
                    # paths_list = data[elem.parameters.cols_names[0]].to_list()

            if 'dataframe' in self.tags.values():
                temp_paths_list = paths_list
            else:
                temp_paths_list = [os.path.join(self.source_path, x) for x in paths_list]
            instr = getattr(CreateArray(), f"instructions_{decamelize(elem.type)}")(temp_paths_list, **elem.native())

            if not elem.type in [LayerOutputTypeChoice.Classification, LayerInputTypeChoice.Dataframe]:
                y_classes = sorted(list(instr['instructions'].keys())) if \
                    isinstance(instr['instructions'], dict) else instr['instructions']
                self.y_cls = [os.path.basename(os.path.dirname(dir_name)) for dir_name in y_classes]

            instructions_data = InstructionsData(
                **getattr(CreateArray(), f"cut_{decamelize(elem.type)}")(instr['instructions'], self.temp_directory,
                                                                         os.path.join(self.paths.sources,
                                                                                      f"{elem.id}_{decamelize(elem.type)}"),
                                                                         **instr['parameters']))
            if 'dataframe' in self.tags.values():
                pass
            else:
                if elem.type not in [LayerInputTypeChoice.Text, LayerOutputTypeChoice.Text,
                                     LayerOutputTypeChoice.TextSegmentation, LayerOutputTypeChoice.Regression]:
                    if elem.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Image,
                                     LayerOutputTypeChoice.Segmentation]:
                        new_paths = [os.path.join('sources', f'{elem.id}_{decamelize(elem.type)}',
                                                  path.replace(self.source_directory + os.path.sep, '')) for path in
                                     instructions_data.instructions]
                    else:
                        new_paths = [os.path.join('sources', path.replace(self.temp_directory + os.path.sep, '')) for
                                     path
                                     in instructions_data.instructions]
                    instructions_data.instructions = new_paths

            instructions.update([(elem.id, instructions_data)])

        return instructions

    def create_preprocessing(self, instructions: DatasetInstructionsData):

        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            if 'MinMaxScaler' in put.parameters.keys() or 'length' in put.parameters.keys():
                self.preprocessing.create_scaler(put.parameters['put'], array=put.instructions, **put.parameters)
            elif 'scaler' in put.parameters.keys() and put.parameters['scaler'] != LayerScalerImageChoice.no_scaler:
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

        for elem in creation_data.outputs:
            if elem.type == LayerOutputTypeChoice.Timeseries:
                classes = list(self.instructions.outputs.get(elem.id).instructions.values())[0]
            else:
                classes = self.instructions.outputs.get(elem.id).instructions
            if elem.type == LayerOutputTypeChoice.Classification:
                if creation_data.outputs.get(2).parameters.type_processing == 'ranges':
                    dfr = pd.read_csv(
                        creation_data.outputs.get(2).parameters.sources_paths[0],
                        usecols=creation_data.outputs.get(2).parameters.cols_names,
                        sep=None, engine='python')
                    dfr.sort_values(by=dfr.columns[0],
                                    ignore_index=True,
                                    inplace=True)
                    column = dfr.values
                    ranges = creation_data.outputs.get(2).parameters.ranges
                    if len(ranges) == 1:
                        border = int(max(column)) / int(ranges)
                        classes_names = np.linspace(border, int(max(column)),
                                                    int(ranges)).tolist()
                    else:
                        classes_names = ranges.split(" ")
                    classes = []
                    for value in column:
                        for i, cl_name in enumerate(classes_names):
                            if value <= int(cl_name):
                                classes.append(i)
                                break

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
        for key in self.instructions.inputs.keys():
            if self.tags[key] == 'dataframe':
                for i in self.instructions.inputs[key].instructions.keys():
                    build_dataframe[i] = self.instructions.inputs[key].instructions[i].values()
            else:
                build_dataframe[f'{key}_{self.tags[key]}'] = self.instructions.inputs[key].instructions
        for key in self.instructions.outputs.keys():
            if self.tags[key] == 'timeseries':
                for i in self.instructions.outputs[key].instructions.keys():
                    build_dataframe[i] = self.instructions.outputs[key].instructions[i].values()
            else:
                build_dataframe[f'{key}_{self.tags[key]}'] = self.instructions.outputs[key].instructions

        dataframe = pd.DataFrame(build_dataframe)
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)

    def create_input_parameters(self, creation_data: CreationData) -> dict:

        creating_inputs_data = {}
        for key in self.instructions.inputs.keys():
            classes_names = [os.path.basename(x) for x in creation_data.inputs.get(key).parameters.sources_paths]
            num_classes = len(classes_names)
            if creation_data.inputs.get(key).type == LayerInputTypeChoice.Text:
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
                    **self.instructions.inputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])

            elif creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe \
                    and creation_data.inputs.get(key).parameters.length:
                length = creation_data.inputs.get(key).parameters.length
                cols = creation_data.inputs.get(key).parameters.cols_names
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].iloc[range(0, length), :len(cols)].values,
                    **self.instructions.inputs.get(key).parameters,
                    **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])

            elif creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe:
                tmp_cols = creation_data.inputs.get(key).parameters.cols_names
                cols = len(tmp_cols) if tmp_cols else creation_data.inputs.get(key).parameters.example_length
                cols = creation_data.inputs.get(key).parameters.xlen if creation_data.inputs.get(
                    key).parameters.xlen else cols
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].iloc[0, :cols].values,
                    **self.instructions.inputs.get(key).parameters,
                    **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])

            else:
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    os.path.join(self.paths.basepath, self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}']),
                    **self.instructions.inputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])
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
            if (creation_data.outputs.get(key).type in
                [LayerOutputTypeChoice.Text, LayerOutputTypeChoice.TextSegmentation]) or (
                    creation_data.outputs.get(key).type in [LayerOutputTypeChoice.Classification] and
                    creation_data.inputs.get(1).type == LayerInputTypeChoice.Dataframe):
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}'],
                    **self.instructions.outputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])
                cl_names = self.instructions.outputs.get(key).parameters['classes_names']
                classes_names = cl_names if cl_names else [os.path.basename(x) for x in creation_data.outputs.get(
                    key).parameters.sources_paths]
                num_classes = len(classes_names)

            elif creation_data.outputs.get(key).type in [LayerOutputTypeChoice.Timeseries]:
                trend = creation_data.outputs.get(key).parameters.trend
                length = creation_data.outputs.get(key).parameters.length
                depth = creation_data.outputs.get(key).parameters.depth
                ycols = creation_data.outputs.get(key).parameters.cols_names
                tmp_df = pd.read_csv(creation_data.outputs.get(key).parameters.sources_paths[0],
                                     sep=None, engine='python', nrows=1, usecols=ycols)
                or_cols = tmp_df.columns.tolist()
                table_cols = self.dataframe['test'].columns.tolist()
                idxs = []
                for col in or_cols:
                    idxs.append(table_cols.index(col))
                if trend:
                    arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                        [self.dataframe['test'].iloc[0, idxs].values,
                         self.dataframe['test'].iloc[length, idxs].values],
                        **self.instructions.outputs.get(key).parameters,
                        **self.preprocessing.preprocessing.get(key))
                else:
                    arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                        self.dataframe['test'].iloc[range(length, length + depth), idxs].values,
                        **self.instructions.outputs.get(key).parameters,
                        **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                               **arr['parameters'])
                classes_names = None
                num_classes = None

            else:
                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                    os.path.join(self.paths.basepath, self.dataframe['test'].loc[0, f'{key}_{self.tags[key]}']),
                    **self.instructions.outputs.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'], **arr['parameters'])
                cl_names = self.instructions.outputs.get(key).parameters['classes_names']
                classes_names = cl_names if cl_names else [os.path.basename(x) for x in creation_data.outputs.get(
                    key).parameters.sources_paths]
                num_classes = len(classes_names)
            classes_colors = self.instructions.outputs.get(key).parameters.get('classes_colors')
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
                if self.tags[key] == 'object_detection':
                    num_arrays = 6
                    for i in range(num_arrays):
                        globals()[f'current_arrays_{i + 1}'] = []

                elif 'timeseries' in self.tags.values():
                    depth = put_data.get(key).parameters['depth']
                    length = put_data.get(key).parameters['length']
                    step = put_data.get(key).parameters['step']
                    xcols = len(put_data.get(key).parameters['cols_names'])
                    ycols = put_data.get(key).parameters['cols_names']
                    trend = put_data.get(key).parameters['trend']
                    for i in range(0, len(self.dataframe[split]) - length - depth, step):
                        if self.tags[key] == decamelize(LayerInputTypeChoice.Dataframe):
                            arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                self.dataframe[split].iloc[range(i, i + length), :xcols].values,
                                **put_data.get(key).parameters,
                                **self.preprocessing.preprocessing.get(key))
                            array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                           **arr['parameters'])
                        elif self.tags[key] in [decamelize(LayerOutputTypeChoice.Timeseries)]:
                            tmp_df = pd.read_csv(put_data.get(key).parameters['sources_paths'][0],
                                                 sep=None, engine='python', nrows=1, usecols=ycols)
                            or_cols = tmp_df.columns.tolist()
                            table_cols = self.dataframe[split].columns.tolist()
                            idxs = []
                            for col in or_cols:
                                idxs.append(table_cols.index(col))
                            if trend:
                                arr = getattr(CreateArray(), f'create_{self.tags[key]}')([
                                    self.dataframe[split].iloc[i, idxs],
                                    self.dataframe[split].iloc[i + length, idxs]],
                                    **put_data.get(key).parameters,
                                    **self.preprocessing.preprocessing.get(key))
                                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                               **arr['parameters'])
                            else:
                                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                    self.dataframe[split].iloc[range(i + length, i + length + depth), idxs].values,
                                    **put_data.get(key).parameters,
                                    **self.preprocessing.preprocessing.get(key))
                                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                               **arr['parameters'])
                        current_arrays.append(array)
                else:
                    for i in range(len(self.dataframe[split])):
                        if self.tags[key] in [decamelize(LayerInputTypeChoice.Text),
                                              decamelize(LayerOutputTypeChoice.Text),
                                              decamelize(LayerOutputTypeChoice.TextSegmentation)]:
                            arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'],
                                **put_data.get(key).parameters,
                                **self.preprocessing.preprocessing.get(key))
                            array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                           **arr['parameters'])
                        elif 'dataframe' in self.tags.values():
                            if self.tags[key] in [decamelize(LayerInputTypeChoice.Dataframe)]:
                                tmp_cols = put_data.get(key).parameters['cols_names']
                                cols = len(tmp_cols) if tmp_cols else put_data.get(key).parameters['example_length']
                                cols = put_data.get(key).parameters['xlen'] if put_data.get(key).parameters[
                                    'xlen'] else cols
                                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                    self.dataframe[split].iloc[i, :cols].values,
                                    **put_data.get(key).parameters,
                                    **self.preprocessing.preprocessing.get(key))
                                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                               **arr['parameters'])
                            else:
                                arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                    self.dataframe[split].loc[i, f'{key}_{self.tags[key]}'],
                                    **put_data.get(key).parameters, **self.preprocessing.preprocessing.get(key))
                                array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                               **arr['parameters'])
                        else:
                            arr = getattr(CreateArray(), f'create_{self.tags[key]}')(
                                os.path.join(self.paths.basepath,
                                             self.dataframe[split].loc[i, f'{key}_{self.tags[key]}']),
                                **put_data.get(key).parameters,
                                **self.preprocessing.preprocessing.get(key))
                            array = getattr(CreateArray(), f'preprocess_{self.tags[key]}')(arr['instructions'],
                                                                                           **arr['parameters'])
                        current_arrays.append(array)
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
