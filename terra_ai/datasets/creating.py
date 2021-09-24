from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerPrepareMethodChoice, LayerScalerImageChoice, ColumnProcessingTypeChoice, \
    LayerTypeProcessingClassificationChoice
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
        self.source_path = creation_data.source_path
        self.y_cls: list = []
        self.columns = {}

        self.columns_processing = {}
        if creation_data.columns_processing:
            for key, value in creation_data.columns_processing.items():
                self.columns_processing[key] = value

        self.instructions: DatasetInstructionsData = self.create_instructions(creation_data)
        self.create_preprocessing(self.instructions)
        self.create_table(creation_data=creation_data)

        self.inputs = self.create_input_parameters(creation_data=creation_data)
        self.outputs = self.create_output_parameters(creation_data=creation_data)

        if not creation_data.use_generator:
            x_array = self.create_dataset_arrays(put_data=self.instructions.inputs)
            y_array = self.create_dataset_arrays(put_data=self.instructions.outputs)
            self.write_arrays(x_array, y_array)

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
            elif out.type == LayerOutputTypeChoice.ObjectDetection:
                with open(creation_data.source_path.joinpath('obj.names'), 'r') as names:
                    names_list = names.read()
                names_list = [elem for elem in names_list.split('\n') if elem]
                out.parameters.classes_names = names_list
                out.parameters.num_classes = len(names_list)

        if creation_data.columns_processing:
            for worker_name, worker_params in creation_data.columns_processing.items():
                if creation_data.columns_processing[worker_name].type == 'Segmentation':
                    for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type == 'Image':
                            creation_data.columns_processing[worker_name].parameters.height = \
                                creation_data.columns_processing[w_name].parameters.height
                            creation_data.columns_processing[worker_name].parameters.width = \
                                creation_data.columns_processing[w_name].parameters.width
                elif creation_data.columns_processing[worker_name].type == 'Timeseries':
                    for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type == 'Classification':
                            creation_data.columns_processing[w_name].parameters.length = \
                                creation_data.columns_processing[worker_name].parameters.length
                            creation_data.columns_processing[w_name].parameters.depth = \
                                creation_data.columns_processing[worker_name].parameters.depth
                            creation_data.columns_processing[w_name].parameters.step = \
                                creation_data.columns_processing[worker_name].parameters.step
                    # for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type == 'Scaler':
                            creation_data.columns_processing[w_name].parameters.length = \
                                creation_data.columns_processing[worker_name].parameters.length
                            creation_data.columns_processing[w_name].parameters.depth = \
                                creation_data.columns_processing[worker_name].parameters.depth
                            creation_data.columns_processing[w_name].parameters.step = \
                                creation_data.columns_processing[worker_name].parameters.step

        return creation_data

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:

        if creation_data.columns_processing:
            inputs = self.create_dataframe_put_instructions(data=creation_data.inputs)
            outputs = self.create_dataframe_put_instructions(data=creation_data.outputs)
        else:
            inputs = self.create_put_instructions(data=creation_data.inputs)
            outputs = self.create_put_instructions(data=creation_data.outputs)
            for out in creation_data.outputs:
                if out.type == LayerOutputTypeChoice.Classification and self.y_cls:
                    for col_name, data in outputs[out.id].items():
                        data.instructions = self.y_cls

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_dataframe_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]):

        put_parameters = {}

        for put in data:

            df = pd.read_csv(put.parameters.sources_paths[0], nrows=0, sep=None, engine='python').columns
            output_cols = list(put.parameters.cols_names.keys())
            cols_names_dict = {str_idx: df[int(str_idx)] for str_idx in output_cols}

            self.tags[put.id] = {}
            put_columns = {}
            cols_names = list(put.parameters.cols_names.keys())
            dataframe = pd.read_csv(put.parameters.sources_paths[0], usecols=[cols_names_dict[str_idx]
                                                                              for str_idx in cols_names],
                                    sep=None, engine='python')
            for idx, name_index in enumerate(cols_names):
                name = cols_names_dict[name_index]
                instructions_data = None
                for worker in put.parameters.cols_names[name_index]:  # На будущее после 1 октября - очень аккуратно!
                    self.tags[put.id][f'{put.id}_{name}'] = decamelize(self.columns_processing[str(worker)].type)
                    list_of_data = dataframe.loc[:, name].to_numpy().tolist()

                    instr = getattr(CreateArray(),
                                    f'instructions_{decamelize(self.columns_processing[str(worker)].type)}')(
                        list_of_data, **{'cols_names': f'{put.id}_{name}', 'put': put.id},
                        **self.columns_processing[str(worker)].parameters.native())
                    path_flag = False
                    if self.columns_processing[str(worker)].type in [LayerInputTypeChoice.Image,
                                                                     LayerOutputTypeChoice.Image,
                                                                     LayerInputTypeChoice.Video,
                                                                     LayerOutputTypeChoice.Segmentation,
                                                                     LayerInputTypeChoice.Audio,
                                                                     LayerOutputTypeChoice.Audio,
                                                                     LayerOutputTypeChoice.ObjectDetection]:
                        paths_list = [os.path.join(self.source_path, elem) for elem in instr['instructions']]
                        path_flag = True
                    else:
                        paths_list = instr['instructions']
                    instructions_data = InstructionsData(
                        **getattr(CreateArray(),
                                  f"cut_{decamelize(self.columns_processing[str(worker)].type)}")(
                            paths_list, self.temp_directory, os.path.join(self.paths.sources, f'{put.id}_{name}'),
                            **instr['parameters']))
                    if path_flag:
                        instructions_data.instructions = [os.path.join('sources',
                                                                       instructions_data.parameters['cols_names'],
                                                                       path.replace(str(self.source_path) +
                                                                                    os.path.sep, ''))
                                                          for path in instructions_data.instructions]
                put_columns[f'{put.id}_{name}'] = instructions_data
            put_parameters[put.id] = put_columns

        return put_parameters

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:

        put_parameters: dict = {}
        for put in data:
            self.tags[put.id] = {f"{put.id}_{decamelize(put.type)}": decamelize(put.type)}
            paths_list: list = []
            for paths in put.parameters.sources_paths:
                if paths.is_dir():
                    for directory, folder, file_name in sorted(os.walk(os.path.join(self.source_directory, paths))):
                        if file_name:
                            file_folder = directory.replace(self.source_directory, '')[1:]
                            for name in sorted(file_name):
                                paths_list.append(os.path.join(file_folder, name))

            put.parameters.cols_names = f'{put.id}_{decamelize(put.type)}'
            put.parameters.put = put.id
            temp_paths_list = [os.path.join(self.source_path, x) for x in paths_list]
            instr = getattr(CreateArray(), f"instructions_{decamelize(put.type)}")(temp_paths_list,
                                                                                   **put.parameters.native())

            if not put.type == LayerOutputTypeChoice.Classification:
                y_classes = sorted(list(instr['instructions'].keys())) if isinstance(instr['instructions'], dict) else\
                    instr['instructions']
                self.y_cls = [os.path.basename(os.path.dirname(dir_name)) for dir_name in y_classes]

            instructions_data = InstructionsData(
                **getattr(CreateArray(), f"cut_{decamelize(put.type)}")(
                    instr['instructions'], self.temp_directory, os.path.join(self.paths.sources,
                                                                             f"{put.id}_{decamelize(put.type)}"),
                    **instr['parameters']))

            if put.type not in [LayerInputTypeChoice.Text, LayerOutputTypeChoice.Text,
                                LayerOutputTypeChoice.TextSegmentation, LayerOutputTypeChoice.Regression]:
                if put.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Segmentation,
                                LayerOutputTypeChoice.Image, LayerOutputTypeChoice.ObjectDetection]:
                    new_paths = [os.path.join('sources', f'{put.id}_{decamelize(put.type)}',
                                              path.replace(self.source_directory + os.path.sep, '')) for path in
                                 instructions_data.instructions]
                else:
                    new_paths = [os.path.join('sources', path.replace(self.temp_directory + os.path.sep, '')) for
                                 path in instructions_data.instructions]
                instructions_data.instructions = new_paths

            put_parameters[put.id] = {f'{put.id}_{decamelize(put.type)}': instructions_data}

        return put_parameters

    def create_preprocessing(self, instructions: DatasetInstructionsData):

        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            for col_name, data in put.items():
                if 'scaler' in data.parameters.keys():
                    if data.parameters['scaler'] != LayerScalerImageChoice.no_scaler:
                        if 'height' in data.parameters.keys():
                            self.preprocessing.create_scaler(array=None, **data.parameters)
                        else:
                            self.preprocessing.create_scaler(array=put[col_name].instructions, **data.parameters)
                elif 'prepare_method' in data.parameters.keys():
                    if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                             LayerPrepareMethodChoice.bag_of_words]:
                        self.preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
                    elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                        self.preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)

    def create_table(self, creation_data: CreationData):

        classes_dict = {}
        for out in self.instructions.outputs.keys():
            if creation_data.columns_processing.get(str(out)) is not None and\
                    creation_data.columns_processing.get(str(out)).type == LayerOutputTypeChoice.Classification and \
                    creation_data.columns_processing.get(str(out)).parameters.type_processing != \
                    LayerTypeProcessingClassificationChoice.ranges or\
                    creation_data.outputs.get(out).type == LayerOutputTypeChoice.Classification:
                for col_name, data in self.instructions.outputs[out].items():
                    class_names = list(dict.fromkeys(data.instructions))
                    classes_dict = {cl_name: [] for cl_name in class_names}
                    for idx, cl_name in enumerate(data.instructions):
                        classes_dict[cl_name].append(idx)
                break
            else:
                for col_name, data in self.instructions.outputs[out].items():
                    classes_dict = {'one_class': [idx for idx in range(len(data.instructions))]}

        split_sequence = {"train": [], "val": [], "test": []}
        for key, value in classes_dict.items():
            train_len = int(creation_data.info.part.train * len(classes_dict[key]))
            val_len = int(creation_data.info.part.validation * len(classes_dict[key]))

            split_sequence['train'].extend(value[:train_len])
            split_sequence['val'].extend(value[train_len: train_len + val_len])
            split_sequence['test'].extend(value[train_len + val_len:])

        if creation_data.info.shuffle:
            random.shuffle(split_sequence['train'])
            random.shuffle(split_sequence['val'])
            random.shuffle(split_sequence['test'])

        build_dataframe = {}
        for inp in self.instructions.inputs.keys():
            for key, value in self.instructions.inputs[inp].items():
                build_dataframe[key] = value.instructions
                print(len(value.instructions))
        for out in self.instructions.outputs.keys():
            for key, value in self.instructions.outputs[out].items():
                build_dataframe[key] = value.instructions
                print(len(value.instructions))

        dataframe = pd.DataFrame(build_dataframe)
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)
        print(self.dataframe['train'])

    def create_input_parameters(self, creation_data: CreationData) -> dict:

        creating_inputs_data = {}
        path_type_input_list = [LayerInputTypeChoice.Image, LayerInputTypeChoice.Video, LayerInputTypeChoice.Audio]
        for key in self.instructions.inputs.keys():
            input_array = []
            self.columns[key] = {}
            creating_inputs_data[key] = {}
            for col_name, data in self.instructions.inputs[key].items():
                prep = None
                if self.preprocessing.preprocessing.get(key) and\
                        self.preprocessing.preprocessing.get(key).get(col_name):
                    prep = self.preprocessing.preprocessing.get(key).get(col_name)

                if creation_data.inputs.get(key).type in path_type_input_list or\
                        self.columns_processing.get(str(key)) is not None and \
                        self.columns_processing.get(str(key)).type in path_type_input_list:
                    data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                elif 'depth' in data.parameters.keys() and data.parameters['depth']:
                    data_to_pass = data.instructions[0:data.parameters['length']]
                else:
                    data_to_pass = data.instructions[0]

                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass, **data.parameters,
                                                                                   **{'preprocess': prep})

                array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                         **arr['parameters'])
                if not array.shape:
                    array = np.expand_dims(array, 0)
                input_array.append(array)

                classes_names = sorted([os.path.basename(x) for x in creation_data.inputs.get(key).parameters.sources_paths])\
                    if not os.path.isfile(creation_data.inputs.get(key).parameters.sources_paths[0]) else\
                    arr['parameters'].get('classes_names')

                num_classes = len(classes_names) if classes_names else None
                if creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe:
                    column_names = pd.read_csv(creation_data.inputs.get(key).parameters.sources_paths[0], nrows=0,
                                               sep=None, engine='python').columns.to_list()
                    current_col_name = '_'.join(col_name.split('_')[1:])
                    idx = column_names.index(current_col_name)
                    task = creation_data.columns_processing[
                        str(creation_data.inputs.get(key).parameters.cols_names[idx][0])].type
                else:
                    task = creation_data.inputs.get(key).type

                # Прописываем параметры для колонки
                current_column = DatasetInputsData(datatype=DataType.get(len(array.shape), 'DIM'),
                                                   dtype=str(array.dtype),
                                                   shape=array.shape,
                                                   name=creation_data.inputs.get(key).name,
                                                   task=task,
                                                   classes_names=classes_names,
                                                   num_classes=num_classes,
                                                   encoding=LayerEncodingChoice.none
                                                   )
                self.columns[key].update([(col_name, current_column.native())])

            # Прописываем параметры для входа
            timeseries_flag = False
            if creation_data.columns_processing:
                for data in creation_data.columns_processing.values():
                    if data.type == ColumnProcessingTypeChoice.Timeseries:
                        timeseries_flag = True
            input_array = np.concatenate(input_array, axis=0) if not timeseries_flag else np.array(input_array)
            task, classes_colors, classes_names, encoding, num_classes = None, None, None, None, None
            if len(self.columns[key]) == 1:
                for c_name, data in self.columns[key].items():
                    task = data['task']
                    classes_colors = data['classes_colors']
                    classes_names = data['classes_names']
                    num_classes = data['num_classes']
                    encoding = data['encoding']
                    break
            else:
                task = LayerInputTypeChoice.Dataframe
                encoding = LayerEncodingChoice.none
                classes_colors, classes_names, = [], []
                for c_name, data in self.columns[key].items():
                    if data['classes_colors']:
                        classes_colors += data['classes_colors']
                    if data['classes_names']:
                        classes_names += data['classes_names']
                num_classes = len(classes_names) if classes_names else None

            current_input = DatasetInputsData(datatype=DataType.get(len(input_array.shape), 'DIM'),
                                              dtype=str(input_array.dtype),
                                              shape=input_array.shape,
                                              name=creation_data.inputs.get(key).name,
                                              task=task,
                                              classes_colors=classes_colors,
                                              classes_names=classes_names,
                                              num_classes=num_classes,
                                              encoding=encoding
                                              )
            creating_inputs_data[key] = current_input.native()

        return creating_inputs_data

    def create_output_parameters(self, creation_data: CreationData) -> dict:

        creating_outputs_data = {}
        path_type_outputs_list = [LayerOutputTypeChoice.Image, LayerOutputTypeChoice.Segmentation,
                                  LayerOutputTypeChoice.Audio, LayerOutputTypeChoice.ObjectDetection]
        for key in self.instructions.outputs.keys():
            output_array = []
            iters = 1
            for col_name, data in self.instructions.outputs[key].items():
                prep = None
                if self.preprocessing.preprocessing.get(key) and\
                        self.preprocessing.preprocessing.get(key).get(col_name):
                    prep = self.preprocessing.preprocessing.get(key).get(col_name)

                if creation_data.outputs.get(key).type in path_type_outputs_list or\
                        self.columns_processing.get(str(key)) is not None and\
                        self.columns_processing.get(str(key)).type in path_type_outputs_list:
                    data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                elif 'trend' in data.parameters.keys():
                    if data.parameters['trend']:
                        data_to_pass = [data.instructions[0], data.instructions[data.parameters['length']]]
                    else:
                        data_to_pass = data.instructions[data.parameters['length']:data.parameters['length'] +
                                                         data.parameters['depth']]
                else:
                    data_to_pass = data.instructions[0]

                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass, **data.parameters,
                                                                                   **{'preprocess': prep})

                array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                         **arr['parameters'])

                if isinstance(array, list):  # Условие для ObjectDetection
                    output_array = [arr for arr in array]
                else:
                    if not array.shape:
                        array = np.expand_dims(array, 0)
                    output_array.append(array)

                cl_names = data.parameters.get('classes_names')
                classes_names = cl_names if cl_names else\
                    sorted([os.path.basename(x) for x in creation_data.outputs.get(key).parameters.sources_paths])
                num_classes = len(classes_names)

                if creation_data.outputs.get(key).type == LayerOutputTypeChoice.Dataframe:
                    column_names = pd.read_csv(creation_data.outputs.get(key).parameters.sources_paths[0], nrows=0,
                                               sep=None, engine='python').columns.to_list()
                    current_col_name = '_'.join(col_name.split('_')[1:])
                    idx = column_names.index(current_col_name)
                    task = creation_data.columns_processing[
                        str(creation_data.outputs.get(key).parameters.cols_names[idx][0])].type
                else:
                    task = creation_data.outputs.get(key).type

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

                if not creation_data.outputs.get(key).type == LayerOutputTypeChoice.ObjectDetection:
                    array = np.expand_dims(array, 0)

                else:
                    iters = 6
                for i in range(iters):
                    current_output = DatasetOutputsData(datatype=DataType.get(len(array[i].shape), 'DIM'),
                                                        dtype=str(array[i].dtype),
                                                        shape=array[i].shape,
                                                        name=creation_data.outputs.get(key).name,
                                                        task=task,
                                                        classes_names=classes_names,
                                                        classes_colors=classes_colors,
                                                        num_classes=num_classes,
                                                        encoding=encoding
                                                        )
                    self.columns[key + i] = {col_name: current_output.native()}

            if not creation_data.outputs.get(key).type == LayerOutputTypeChoice.ObjectDetection:
                if 'depth' in data.parameters.keys() and data.parameters['depth']:
                    output_array = np.array(output_array)
                else:
                    output_array = np.concatenate(output_array, axis=0)
                    output_array = np.expand_dims(output_array, 0)
            task, classes_colors, classes_names, encoding, num_classes = None, None, None, None, None
            if len(self.columns[key]) == 1:
                for c_name, data in self.columns[key].items():
                    task = data['task']
                    classes_colors = data['classes_colors']
                    classes_names = data['classes_names']
                    num_classes = data['num_classes']
                    encoding = data['encoding']
                    break
            else:
                task = LayerInputTypeChoice.Dataframe
                encoding = LayerEncodingChoice.none
                classes_colors, classes_names, = [], []
                for c_name, data in self.columns[key].items():
                    if data['classes_colors']:
                        classes_colors += data['classes_colors']
                    if data['classes_names']:
                        classes_names += data['classes_names']
                num_classes = len(classes_names) if classes_names else None
            for i in range(iters):
                current_output = DatasetOutputsData(datatype=DataType.get(len(output_array[i].shape), 'DIM'),
                                                    dtype=str(output_array[i].dtype),
                                                    shape=output_array[i].shape,
                                                    name=creation_data.outputs.get(key).name,
                                                    task=task,
                                                    classes_colors=classes_colors,
                                                    classes_names=classes_names,
                                                    num_classes=num_classes,
                                                    encoding=encoding
                                                    )
                creating_outputs_data[key + i] = current_output.native()

        return creating_outputs_data

    def create_dataset_arrays(self, put_data: dict) -> dict:

        path_type_list = [decamelize(LayerInputTypeChoice.Image), decamelize(LayerOutputTypeChoice.Image),
                          decamelize(LayerInputTypeChoice.Audio), decamelize(LayerOutputTypeChoice.Audio),
                          decamelize(LayerInputTypeChoice.Video), decamelize(LayerOutputTypeChoice.ObjectDetection),
                          decamelize(LayerOutputTypeChoice.Segmentation)]

        out_array = {'train': {}, 'val': {}, 'test': {}}
        for split in list(out_array.keys()):
            for key in put_data.keys():
                current_arrays: list = []
                col_name = None

                length, depth, step = 0, 0, 1
                for col_name, data in put_data[key].items():
                    depth = data.parameters['depth'] if 'depth' in data.parameters.keys() and \
                                                        data.parameters['depth'] else 0
                    length = data.parameters['length'] if depth else 0
                    step = data.parameters['step'] if depth else 1

                for j in range(6):
                    globals()[f'current_arrays_{j}'] = []
                for i in range(0, len(self.dataframe[split]) - length - depth, step):
                    full_array = []
                    for col_name, data in put_data[key].items():
                        prep = None
                        if self.tags[key][col_name] in path_type_list:
                            data_to_pass = os.path.join(self.paths.basepath, self.dataframe[split].loc[i, col_name])

                        elif 'depth' in data.parameters.keys() and data.parameters['depth']:
                            if 'trend' in data.parameters.keys() and data.parameters['trend']:
                                data_to_pass = [self.dataframe[split].loc[i, col_name],
                                                self.dataframe[split].loc[i + data.parameters['length'] - 1, col_name]]
                            elif 'trend' in data.parameters.keys():
                                data_to_pass = self.dataframe[split].loc[i + data.parameters['length']:i +
                                                                         data.parameters['length'] +
                                                                         data.parameters['depth'] - 1, col_name]
                            else:
                                data_to_pass = self.dataframe[split].loc[i:i + data.parameters['length'] - 1, col_name]
                        else:
                            data_to_pass = self.dataframe[split].loc[i, col_name]

                        if self.preprocessing.preprocessing.get(key) and\
                                self.preprocessing.preprocessing.get(key).get(col_name):
                            prep = self.preprocessing.preprocessing.get(key).get(col_name)
                        arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass,
                                                                                           **{'preprocess': prep},
                                                                                           **data.parameters)

                        arr = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                               **arr['parameters'])

                        if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            for n in range(6):
                                globals()[f'current_arrays_{n}'].append(arr[n])
                        else:
                            full_array.append(arr)
                    if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                        if depth:
                            array = np.array(full_array)
                        else:
                            array = np.concatenate(full_array, axis=0)
                        current_arrays.append(array)

                if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    for n in range(6):
                        print(np.array(globals()[f'current_arrays_{n}']).shape)
                        out_array[split][key + n] = np.array(globals()[f'current_arrays_{n}'])
                else:
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
                inp = inp.native()
                del inp['parameters']['sources_paths']
                json.dump(inp, cfg)
        for out in creation_data.outputs:
            with open(os.path.join(parameters_path, f'{out.id}_outputs.json'), 'w') as cfg:
                out = out.native()
                del out['parameters']['sources_paths']
                json.dump(out, cfg)

        # if self.columns_processing:
        #     with open(os.path.join(parameters_path, f'0_columns_preprocessing.json'), 'w') as cfg:
        #         inp = inp.native()
        #         del inp['parameters']['sources_paths']
        #         json.dump(inp, cfg)

        os.makedirs(tables_path, exist_ok=True)
        for key in self.dataframe.keys():
            self.dataframe[key].to_csv(os.path.join(self.paths.instructions, 'tables', f'{key}.csv'))

        pass

    def write_preprocesses_to_files(self):

        for put, proc in self.preprocessing.preprocessing.items():
            for col_name, obj in proc.items():
                if obj:
                    folder_dir = os.path.join(self.paths.preprocessing, str(put))
                    os.makedirs(folder_dir, exist_ok=True)
                    joblib.dump(obj, os.path.join(folder_dir, f'{col_name}.gz'))

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

        for attr in ['inputs', 'outputs', 'columns']:
            data[attr] = self.__dict__[attr]

        with open(os.path.join(self.paths.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(DatasetData(**data).native(), fp)
        print(DatasetData(**data).native())

        return data
