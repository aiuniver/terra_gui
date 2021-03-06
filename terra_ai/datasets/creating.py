from terra_ai.utils import decamelize, camelize, autodetect_encoding
from terra_ai.exceptions.tensor_flow import ResourceExhaustedError as Resource
from terra_ai.datasets.data import DataType, InstructionsData, DatasetInstructionsData
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList, \
    ColumnsProcessingData, CreationOutputData, CreationInputData
from terra_ai.data.datasets.dataset import DatasetData, DatasetInputsData, DatasetOutputsData, DatasetPathsData
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerPrepareMethodChoice, LayerScalerImageChoice, ColumnProcessingTypeChoice, \
    LayerTypeProcessingClassificationChoice, LayerEncodingChoice, LayerTransformerMethodChoice
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG
from terra_ai.data.datasets.creations.layers.output.types.ObjectDetection import LayerODDatasetTypeChoice
from terra_ai import progress
from terra_ai.logging import logger

import psutil
import cv2
import os
import random
import numpy as np
import pandas as pd
import json
import joblib
import tempfile
import shutil
import zipfile
import concurrent.futures
import h5py
from math import ceil
from itertools import repeat
from pathlib import Path
from typing import Union
from datetime import datetime
from pytz import timezone


class CreateDataset(object):

    progress_name = 'create_dataset'

    @progress.threading
    def __init__(self, cr_data: CreationData):

        progress.pool.reset(name=self.progress_name,
                            message='Начало',
                            finished=False)
        logger.info(f'Начало формирования датасета {cr_data.name}')
        try:
            creation_data = self.preprocess_creation_data(cr_data)
        except Exception:
            logger.error('Ошибка выбора параметров создания датасета', extra={'type': "warning"})
            progress.pool(self.progress_name,
                          error='Ошибка выбора параметров создания датасета')
            raise
        self.temp_directory = tempfile.mkdtemp()
        os.makedirs(Path(self.temp_directory, f'{creation_data.alias}.{DATASET_EXT}'), exist_ok=True)
        self.paths = DatasetPathsData(basepath=Path(self.temp_directory, f'{creation_data.alias}.{DATASET_EXT}'))

        self.source_directory: str = str(creation_data.source_path)
        self.dataframe: dict = {}
        self.temporary: dict = {}
        self.tags: dict = {}
        self.preprocessing = CreatePreprocessing()
        self.use_generator: bool = False if not creation_data.use_generator else True
        self.source_path = creation_data.source_path
        self.y_cls: list = []
        self.columns = {}
        self.augmentation = {}

        self.columns_processing = {}
        if creation_data.columns_processing:
            for key, value in creation_data.columns_processing.items():
                self.columns_processing[key] = value

        self.instructions: DatasetInstructionsData = self.create_instructions(creation_data)

        progress.pool(self.progress_name, message='Создание препроцессинга', percent=0)
        self.create_preprocessing(self.instructions)
        self.fit_preprocessing(put_data=self.instructions.inputs)
        self.fit_preprocessing(put_data=self.instructions.outputs)
        self.create_table(creation_data=creation_data)

        self.inputs = self.create_input_parameters(creation_data=creation_data)
        self.outputs = self.create_output_parameters(creation_data=creation_data)
        self.service = self.create_service_parameters(creation_data=creation_data)

        if not creation_data.outputs[0].type in [LayerOutputTypeChoice.Speech2Text, LayerOutputTypeChoice.Text2Speech,
                                                 LayerOutputTypeChoice.Tracker]:
            self.create_dataset_arrays(put_data=self.instructions.inputs)
            if not creation_data.outputs[0].type in [LayerOutputTypeChoice.ImageGAN, LayerOutputTypeChoice.ImageCGAN]:
                self.create_dataset_arrays(put_data=self.instructions.outputs)

        self.write_preprocesses_to_files()
        self.write_instructions_to_files()

        progress.pool(self.progress_name,
                      message='Сохранение датасета',
                      percent=100
                      )

        self.zip_dataset(self.paths.basepath, os.path.join(self.temp_directory, 'dataset'))
        shutil.move(os.path.join(self.temp_directory, 'dataset.zip'), self.paths.basepath)
        for key, value in self.paths.__dict__.items():
            if not key == 'basepath':
                shutil.rmtree(value)
        self.datasetdata = DatasetData(**self.write_dataset_configure(creation_data=creation_data))

        if Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}').is_dir():
            shutil.rmtree(Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}'))
        shutil.move(str(self.paths.basepath), creation_data.datasets_path)
        shutil.rmtree(self.temp_directory)

        progress.pool(self.progress_name,
                      percent=100,
                      message='Формирование завершено',
                      finished=True,
                      data=self.datasetdata
                      )
        logger.info(f'Создан датасет {creation_data.name}', extra={'type': "info"})

    @staticmethod
    def postprocess_timeseries(full_array):
        try:
            new_array = np.array(full_array).transpose()
        except:
            new_array = []
            array = []
            for el in full_array:
                if type(el[0]) == np.ndarray:
                    tmp = []
                    for j in range(len(el)):
                        tmp.append(list(el[j]))
                    array.append(tmp)
                else:
                    array.append(el.tolist())
            array = np.array(array).transpose().tolist()
            for i in array:
                tmp = []
                for j in i:
                    if type(j) == list:
                        tmp.extend(j)
                    else:
                        tmp.append(j)
                new_array.append(tmp)
            new_array = np.array(new_array)
        return new_array

    @staticmethod
    def preprocess_creation_data(creation_data):

        noise_flag = False
        for inp in range(len(creation_data.inputs)):
            for worker_name, worker_params in creation_data.columns_processing.items():
                if creation_data.columns_processing[worker_name].type == 'Image':
                    shape = (worker_params.parameters.height, worker_params.parameters.width, 3)
                elif creation_data.columns_processing[worker_name].type == 'ImageGAN' and not noise_flag:
                    new_worker_id = int(list(creation_data.columns_processing.keys())[-1]) + 1
                    creation_data.columns_processing[str(new_worker_id)] = ColumnsProcessingData(
                        **{"type": "Noise", "parameters": {'shape': (100,)}})
                    output_copy = creation_data.outputs[0]
                    creation_data.inputs.append(CreationInputData(
                        **{'id': creation_data.inputs[-1].id + 1,
                           'name': 'Шум',
                           'type': LayerOutputTypeChoice.Dataframe,
                           'parameters': {
                               'cols_names': {list(output_copy.parameters.cols_names.keys())[0]: [new_worker_id]},
                               'sources_paths': output_copy.parameters.sources_paths}})
                    )
                    noise_flag = True
                    creation_data.use_generator = True
                    break

        for out in creation_data.outputs:
            if out.type in [LayerOutputTypeChoice.Classification, LayerOutputTypeChoice.Tracker,
                            LayerOutputTypeChoice.Speech2Text, LayerOutputTypeChoice.Text2Speech]:
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
                        filters = inp.parameters.filters
                        for x in out.parameters.open_tags + out.parameters.close_tags:
                            filters = filters.replace(x, '')
                        inp.parameters.filters = filters
                        out.parameters.filters = filters
                        inp.parameters.open_tags = out.parameters.open_tags
                        inp.parameters.close_tags = out.parameters.close_tags
            elif out.type == LayerOutputTypeChoice.ObjectDetection:
                for inp in creation_data.inputs:
                    if inp.type == LayerInputTypeChoice.Image:
                        out.parameters.frame_mode = inp.parameters.image_mode
                names_list = get_od_names(creation_data)
                out.parameters.classes_names = names_list
                out.parameters.num_classes = len(names_list)
            elif out.type in [LayerOutputTypeChoice.ImageGAN, LayerOutputTypeChoice.ImageCGAN]:
                creation_data.use_generator = True
                out_list = []
                img_shape = ()
                sources_paths = []
                for inp in creation_data.inputs:
                    if inp.type == LayerInputTypeChoice.Image:
                        img_shape = (inp.parameters.height, inp.parameters.width, 3)
                        sources_paths = inp.parameters.sources_paths
                idx = 2
                creation_data.inputs.append(
                    CreationInputData(
                        id=idx,
                        name='Шум',
                        type=LayerInputTypeChoice.Noise,
                        parameters={'sources_paths': sources_paths,
                                    'shape': (100,)}))
                idx += 1
                if out.type == LayerOutputTypeChoice.ImageCGAN:
                    creation_data.inputs.append(
                        CreationInputData(
                            id=idx,
                            name='Классы',
                            type=LayerInputTypeChoice.Classification,
                            parameters={'sources_paths': sources_paths,
                                        'type_processing': 'categorical'}))
                    idx += 1
                    creation_data.inputs.append(
                        CreationInputData(
                            id=idx,
                            name='Классы',
                            type=LayerInputTypeChoice.Classification,
                            parameters={'sources_paths': sources_paths,
                                        'type_processing': 'categorical'}))
                    idx += 1
                out_list.append(
                    CreationOutputData(
                        id=idx,
                        name='Генератор',
                        type=LayerOutputTypeChoice.Generator,
                        parameters={'sources_paths': sources_paths,
                                    'shape': img_shape}).native())
                idx += 1
                out_list.append(
                    CreationOutputData(
                        id=idx,
                        name='Дискриминатор',
                        type=LayerOutputTypeChoice.Discriminator,
                        parameters={'sources_paths': sources_paths,
                                    'shape': (1,)}).native())
                creation_data.outputs = CreationOutputsList(out_list)

        if creation_data.columns_processing:
            worker_keys = list(creation_data.columns_processing.keys())
            for worker_name in worker_keys:
                if creation_data.columns_processing[worker_name].type == 'Segmentation':
                    for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type == 'Image':
                            creation_data.columns_processing[worker_name].parameters.height = \
                                creation_data.columns_processing[w_name].parameters.height
                            creation_data.columns_processing[worker_name].parameters.width = \
                                creation_data.columns_processing[w_name].parameters.width
                elif creation_data.columns_processing[worker_name].type == 'Timeseries':
                    if creation_data.columns_processing[worker_name].parameters.trend:
                        creation_data.columns_processing[worker_name].parameters.depth = 1
                    for w_name, w_params in creation_data.columns_processing.items():
                        if creation_data.columns_processing[w_name].type in ['Classification', 'Scaler']:
                            creation_data.columns_processing[w_name].parameters.length = \
                                creation_data.columns_processing[worker_name].parameters.length
                            creation_data.columns_processing[w_name].parameters.depth = \
                                creation_data.columns_processing[worker_name].parameters.depth
                            creation_data.columns_processing[w_name].parameters.step = \
                                creation_data.columns_processing[worker_name].parameters.step
                elif creation_data.columns_processing[worker_name].type in [LayerOutputTypeChoice.ImageGAN.name,
                                                                            LayerOutputTypeChoice.ImageCGAN.name]:
                    creation_data.use_generator = True
                    new_worker_id = int(list(creation_data.columns_processing.keys())[-1]) + 1
                    creation_data.columns_processing[worker_name] = ColumnsProcessingData(
                        **{"type": "Generator", "parameters": {'shape': shape}})
                    creation_data.columns_processing[str(new_worker_id)] = ColumnsProcessingData(
                        **{"type": "Discriminator", "parameters": {'shape': (1,)}})
                    creation_data.outputs[0].id = creation_data.inputs[-1].id + 1
                    creation_data.outputs[0].name = 'Генератор'
                    output_copy = creation_data.outputs[0]
                    creation_data.outputs.append(CreationOutputData(
                        **{'id': creation_data.outputs[-1].id + 1,
                           'name': 'Дискриминатор',
                           'type': LayerOutputTypeChoice.Dataframe,
                           'parameters': {'cols_names': {
                               list(output_copy.parameters.cols_names.keys())[0]: [new_worker_id]},
                                          'sources_paths': output_copy.parameters.sources_paths}})
                    )
                    break

                elif creation_data.columns_processing[worker_name].type == 'Transformer':
                    creation_data.columns_processing['0'].parameters.transformer = LayerTransformerMethodChoice.enc_inp
                    creation_data.columns_processing['1'] = ColumnsProcessingData(
                        **creation_data.columns_processing['0'].native().copy())
                    creation_data.columns_processing['1'].parameters.transformer = LayerTransformerMethodChoice.dec_inp
                    creation_data.columns_processing['2'] = ColumnsProcessingData(
                        **creation_data.columns_processing['0'].native().copy())
                    creation_data.columns_processing['2'].parameters.transformer = LayerTransformerMethodChoice.dec_out
                    original = creation_data.inputs[0].native()
                    creation_data.inputs.append(CreationInputData(
                        id=original['id'] + 1,
                        name='Декодер',
                        type=original['type'],
                        parameters={
                            'sources_paths': original['parameters']['sources_paths'],
                            'cols_names': {'1': [1]}}
                    ))
                    creation_data.outputs[0].id += 1
                    for key in creation_data.outputs[0].parameters.cols_names.keys():
                        creation_data.outputs[0].parameters.cols_names[key] = [2]

        return creation_data

    def create_instructions(self, creation_data: CreationData) -> DatasetInstructionsData:

        if creation_data.columns_processing:
            inputs = self.create_dataframe_put_instructions(data=creation_data.inputs)
            outputs = self.create_dataframe_put_instructions(data=creation_data.outputs)
        else:
            inputs = self.create_put_instructions(data=creation_data.inputs)
            outputs = self.create_put_instructions(data=creation_data.outputs)
            for inp in creation_data.inputs:
                if inp.type == LayerInputTypeChoice.Classification and self.y_cls:
                    for col_name, data in inputs[inp.id].items():
                        data.instructions = self.y_cls
            for out in creation_data.outputs:
                if out.type == LayerOutputTypeChoice.Classification and self.y_cls:
                    for col_name, data in outputs[out.id].items():
                        data.instructions = self.y_cls
                elif out.type in [LayerOutputTypeChoice.Tracker, LayerOutputTypeChoice.Text2Speech,
                                  LayerOutputTypeChoice.Speech2Text] and self.y_cls:
                    for col_name, data in outputs[out.id].items():
                        data.instructions = ['no_data' for _ in self.y_cls]

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_dataframe_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]):

        def instructions(item, cur_put, proc_worker):
            try:
                instr = getattr(CreateArray(),
                                f'instructions_{decamelize(self.columns_processing[str(proc_worker)].type)}')(
                    [item], **{'cols_names': f'{put.id}_{name}', 'put': put.id},
                    **self.columns_processing[str(proc_worker)].parameters.native())
                cut_data = getattr(CreateArray(),
                                   f"cut_{decamelize(self.columns_processing[str(proc_worker)].type)}")(
                    instr['instructions'],
                    self.paths.sources.joinpath(f"{cur_put.id}_", f"{decamelize(cur_put.type)}"),
                    **instr['parameters']
                )
            except Exception:
                progress.pool(self.progress_name, error='Ошибка создания инструкций')
                logger.exception('Ошибка создания инструкций', extra={'type': "warning"})
                raise

            return cut_data

        put_parameters = {}

        for put in data:
            try:
                _, enc = autodetect_encoding(put.parameters.sources_paths[0], True)
                df = pd.read_csv(put.parameters.sources_paths[0], nrows=0, sep=None,
                                 engine='python', encoding=enc).columns
            except Exception:
                logger.exception('Ошибка чтения csv-файла', extra={'type': "warning"})
                progress.pool(self.progress_name, error='Ошибка чтения csv-файла')
                raise
            output_cols = list(put.parameters.cols_names.keys())
            cols_names_dict = {str_idx: df[int(str_idx)] for str_idx in output_cols}

            self.tags[put.id] = {}
            put_columns = {}
            cols_names = list(put.parameters.cols_names.keys())
            _, enc = autodetect_encoding(put.parameters.sources_paths[0], True)
            dataframe = pd.read_csv(put.parameters.sources_paths[0], usecols=[cols_names_dict[str_idx]
                                                                              for str_idx in cols_names],
                                    sep=None, engine='python', encoding=enc)
            for idx, name_index in enumerate(cols_names):
                name = cols_names_dict[name_index]
                instructions_data = None
                list_of_data = dataframe.loc[:, name].to_numpy().tolist()
                if put.parameters.cols_names[name_index]:
                    try:
                        for worker in put.parameters.cols_names[name_index]:  # На будущее после 1 октября - очень аккуратно!
                            self.tags[put.id][f'{put.id}_{name}'] = decamelize(self.columns_processing[str(worker)].type)
                            if decamelize(self.columns_processing[str(worker)].type) in PATH_TYPE_LIST:
                                # list_of_data = [os.path.join(self.source_path, x) for x in list_of_data]
                                list_of_data = [Path(self.source_path).joinpath(Path(x)) for x in list_of_data]
                            results_list = []
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                results = executor.map(instructions, list_of_data, repeat(put), repeat(worker))
                                progress.pool(self.progress_name,
                                              message=f'Обработка колонки {name}')
                                for i, result in enumerate(results):
                                    progress.pool(self.progress_name, percent=ceil(i / len(list_of_data) * 100))
                                    if decamelize(self.columns_processing[str(worker)].type) in PATH_TYPE_LIST:
                                        new_paths = [path.replace(str(self.paths.basepath) + os.path.sep, '')
                                                     for path in result['instructions']]
                                        result['instructions'] = new_paths
                                    results_list += result['instructions']
                            result['parameters'].update([('put_type', decamelize(self.columns_processing[str(worker)].type))])
                            instructions_data = InstructionsData(instructions=results_list,
                                                                 parameters=result['parameters'])
                            if instructions_data.parameters['put_type'] == 'classification':
                                list_of_classes = []
                                for x in list_of_data:
                                    if x not in list_of_classes:
                                        list_of_classes.append(x)
                                instructions_data.parameters['classes_names'] = list_of_classes
                                instructions_data.parameters['num_classes'] = len(list_of_classes)
                    except Exception:
                        progress.pool(self.progress_name, error='Ошибка создания инструкций')
                        logger.exception('Ошибка создания инструкций', extra={'type': "warning"})
                        raise
                else:
                    self.tags[put.id][f'{put.id}_{name}'] = decamelize(LayerInputTypeChoice.Raw)
                    instructions_data = InstructionsData(**{'instructions': list_of_data,
                                                            'parameters': {'put_type': decamelize(
                                                                LayerInputTypeChoice.Raw),
                                                                'put': put.id,
                                                                'cols_names': f'{put.id}_{name}'
                                                            }
                                                            }
                                                         )

                put_columns[f'{put.id}_{name}'] = instructions_data
            put_parameters[put.id] = put_columns

        return put_parameters

    def create_put_instructions(self, data: Union[CreationInputsList, CreationOutputsList]) -> dict:

        def instructions(path, put):
            try:
                instr = getattr(CreateArray(), f"instructions_{decamelize(put.type)}")([path],
                                                                                       **put.parameters.native())
                cut_data = getattr(CreateArray(), f"cut_{decamelize(put.type)}")(instr['instructions'],
                                                                                 os.path.join(self.paths.sources,
                                                                                              f"{put.id}_"
                                                                                              f"{decamelize(put.type)}"),
                                                                                 **instr['parameters'])
            except Exception:
                progress.pool(self.progress_name, error='Ошибка создания инструкций')
                logger.exception('Ошибка создания инструкций', extra={'type': "warning"})
                raise

            class_name = [os.path.basename(os.path.dirname(x)) for x in list(instr['instructions'].keys())]\
                if put.type == LayerInputTypeChoice.Text else None

            if class_name:
                return cut_data, class_name
            else:
                return (cut_data, )

        put_parameters: dict = {}
        for put in data:
            self.tags[put.id] = {f"{put.id}_{decamelize(put.type)}": decamelize(put.type)}
            if self.tags.get(put.id - 1) is not None and self.tags.get(put.id - 1).get(f"{put.id - 1}_"
                                                                                       f"{decamelize(put.type)}") == \
                    decamelize(LayerInputTypeChoice.Audio):
                instructions = put_parameters[put.id - 1][f'{put.id - 1}_{decamelize(put.type)}'].instructions.copy()
                parameters = put_parameters[put.id - 1][f'{put.id - 1}_{decamelize(put.type)}'].parameters.copy()
                parameters['parameter'] = put.parameters.parameter
                parameters['cols_names'] = f'{put.id}_{decamelize(put.type)}'
                parameters['put'] = put.id
                instructions_data = InstructionsData(instructions=instructions, parameters=parameters)
            else:
                paths_list: list = []
                if 'model_type' in put.parameters.native().keys() and \
                                                        put.parameters.model_type in [LayerODDatasetTypeChoice.Udacity]:
                    for file_name in os.listdir(os.sep.join(str(put.parameters.sources_paths).split(os.sep)[:-1])):
                        if file_name.endswith('.csv'):
                            paths_list.append(file_name)

                elif 'model_type' in put.parameters.native().keys() and \
                        put.parameters.model_type in [LayerODDatasetTypeChoice.Yolov1]:
                    for paths in put.parameters.sources_paths:
                        if paths.is_dir():
                            for directory, folder, file_name in sorted(os.walk(os.path.join(self.source_directory,
                                                                                            paths))):
                                if file_name:
                                    file_folder = directory.replace(self.source_directory, '')[1:]
                                    for name in sorted(file_name):
                                        if name.endswith('.txt'):
                                            paths_list.append(os.path.join(file_folder, name))

                elif decamelize(put.type) == decamelize(LayerInputTypeChoice.Image):
                    for paths in put.parameters.sources_paths:
                        if paths.is_dir():
                            for directory, folder, file_name in sorted(os.walk(os.path.join(self.source_directory,
                                                                                            paths))):
                                if file_name:
                                    file_folder = directory.replace(self.source_directory, '')[1:]
                                    for name in sorted(file_name):
                                        if not name.endswith('.txt'):
                                            paths_list.append(os.path.join(file_folder, name))

                else:
                    for paths in put.parameters.sources_paths:
                        if paths.is_dir():
                            for directory, folder, file_name in sorted(os.walk(os.path.join(self.source_directory,
                                                                                            paths))):
                                if file_name:
                                    file_folder = directory.replace(self.source_directory, '')[1:]
                                    for name in sorted(file_name):
                                        paths_list.append(os.path.join(file_folder, name))
                put.parameters.cols_names = f'{put.id}_{decamelize(put.type)}'
                put.parameters.put = put.id
                temp_paths_list = [os.path.join(self.source_path, x) for x in paths_list]

                results_list = []
                parameters = {}
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(instructions, temp_paths_list, repeat(put))
                    progress.pool(self.progress_name,
                                  message='Формирование файлов')
                    for i, result in enumerate(results):
                        progress.pool(self.progress_name, percent=ceil(i / len(temp_paths_list) * 100))
                        results_list += result[0]['instructions']
                        parameters = result[0]['parameters']
                        if put.type not in [LayerOutputTypeChoice.Classification, LayerOutputTypeChoice.Segmentation,
                                            LayerOutputTypeChoice.TextSegmentation,
                                            LayerOutputTypeChoice.ObjectDetection, LayerOutputTypeChoice.Timeseries,
                                            LayerOutputTypeChoice.TimeseriesTrend, LayerOutputTypeChoice.Regression,
                                            LayerOutputTypeChoice.Tracker, LayerOutputTypeChoice.Speech2Text,
                                            LayerOutputTypeChoice.Text2Speech, LayerOutputTypeChoice.Generator,
                                            LayerOutputTypeChoice.Discriminator, LayerInputTypeChoice.Noise]:
                            y_classes = result[1] if len(result) > 1 else [os.path.basename(os.path.dirname(dir_name))
                                                                           for dir_name in result[0]['instructions']]
                            self.y_cls += y_classes

                instructions_data = InstructionsData(instructions=results_list, parameters=parameters)
                if decamelize(put.type) in PATH_TYPE_LIST:
                    new_paths = [path.replace(str(self.paths.basepath) + os.path.sep, '')
                                 for path in instructions_data.instructions]
                    instructions_data.instructions = new_paths

                instructions_data.parameters.update([('put_type', decamelize(put.type))])

            put_parameters[put.id] = {f'{put.id}_{decamelize(put.type)}': instructions_data}

        return put_parameters

    def create_preprocessing(self, instructions: DatasetInstructionsData):

        saved_prep = None
        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            for col_name, data in put.items():
                if 'timeseries' in data.parameters.values():
                    length = data.parameters['length']
                    depth = data.parameters['depth']
                    step = data.parameters['step']
                    for pt in list(instructions.inputs.values()) + list(instructions.outputs.values()):
                        for col_nm, dt in pt.items():
                            if 'raw' in dt.parameters.values():
                                dt.parameters['length'] = length
                                dt.parameters['depth'] = depth
                                dt.parameters['step'] = step
                if 'scaler' in data.parameters.keys():
                    self.preprocessing.create_scaler(**data.parameters)
                elif 'prepare_method' in data.parameters.keys():
                    if data.parameters['transformer'] == 'dec_out':
                        self.preprocessing.preprocessing[data.parameters['put']] = {}
                        self.preprocessing.preprocessing[data.parameters['put']].update({data.parameters['cols_names']:
                                                                                             saved_prep})
                    else:
                        if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                                 LayerPrepareMethodChoice.bag_of_words]:
                            self.preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
                        elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                            self.preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)
                        saved_prep = self.preprocessing.preprocessing.get(data.parameters['put']).get(
                            data.parameters['cols_names'])

                # if 'augmentation' in data.parameters.keys() and data.parameters['augmentation']:
                    # self.augmentation[data.parameters['cols_names']] = {'train': [], 'val': []}
                    # {'object': self.preprocessing.create_image_augmentation(data.parameters['augmentation']),
                    # 'data': []}

    def fit_preprocessing(self, put_data):

        for key in put_data.keys():
            for col_name, data in put_data[key].items():
                if 'scaler' in data.parameters and data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler,
                                                                                     None]:
                    progress.pool(self.progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
                    try:
                        if self.tags[key][col_name] in PATH_TYPE_LIST:
                            for i in range(len(data.instructions)):
                                progress.pool(self.progress_name,
                                              percent=ceil(i / len(data.instructions) * 100))

                                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                                    os.path.join(self.paths.basepath, data.instructions[i]),
                                    **data.parameters)

                                if data.parameters['put_type'] in [decamelize(LayerInputTypeChoice.Image),
                                                                   decamelize(LayerOutputTypeChoice.Image)]:
                                    arr = {'instructions': cv2.resize(arr['instructions'], (data.parameters['width'],
                                                                                            data.parameters['height']))}
                                if data.parameters['scaler'] == 'terra_image_scaler':
                                    self.preprocessing.preprocessing[key][col_name].fit(arr['instructions'])
                                else:
                                    self.preprocessing.preprocessing[key][col_name].fit(arr['instructions'].reshape(-1, 1))
                        else:
                            self.preprocessing.preprocessing[key][col_name].fit(np.array(data.instructions).reshape(-1, 1))
                    except Exception:
                        progress.pool(self.progress_name, error='Ошибка обучения скейлера')
                        logger.exception('Ошибка обучения скейлера', extra={'type': "warning"})
                        raise

    def create_table(self, creation_data: CreationData):

        classes_dict = {}
        for out in self.instructions.outputs.keys():
            if creation_data.columns_processing.get(str(out)) is not None and \
                    creation_data.columns_processing.get(str(out)).type == LayerOutputTypeChoice.Classification and \
                    creation_data.columns_processing.get(str(out)).parameters.type_processing != \
                    LayerTypeProcessingClassificationChoice.ranges or \
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

        if creation_data.info.shuffle:
            for key in classes_dict.keys():
                random.shuffle(classes_dict[key])

        split_sequence = {"train": [], "val": []}
        for key, value in classes_dict.items():
            train_len = int(creation_data.info.part.train * len(classes_dict[key]))

            split_sequence['train'].extend(value[:train_len])
            split_sequence['val'].extend(value[train_len:])

        if creation_data.info.shuffle:
            random.shuffle(split_sequence['train'])
            random.shuffle(split_sequence['val'])

        build_dataframe = {}
        for inp in self.instructions.inputs.keys():
            for key, value in self.instructions.inputs[inp].items():
                build_dataframe[key] = value.instructions
                # print(len(value.instructions))
        for out in self.instructions.outputs.keys():
            for key, value in self.instructions.outputs[out].items():
                build_dataframe[key] = value.instructions
                # print(len(value.instructions))
        try:
            dataframe = pd.DataFrame(build_dataframe)
            # print(dataframe)
        except Exception:
            message = 'Ошибка создания датасета. Несоответствие количества входных/выходных данных'
            progress.pool(self.progress_name,
                          error=message)
            logger.exception(message, extra={'type': "warning"})
            raise
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)

    def create_input_parameters(self, creation_data: CreationData) -> dict:

        creating_inputs_data = {}
        for key in self.instructions.inputs.keys():
            input_array = []
            self.columns[key] = {}
            creating_inputs_data[key] = {}
            for col_name, data in self.instructions.inputs[key].items():
                column_names = []
                encoding = LayerEncodingChoice.none
                if creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe:
                    try:
                        _, enc = autodetect_encoding(creation_data.inputs.get(key).parameters.sources_paths[0], True)
                        column_names = pd.read_csv(creation_data.inputs.get(key).parameters.sources_paths[0], nrows=0,
                                                   sep=None, engine='python', encoding=enc).columns.to_list()
                    except Exception:
                        progress.pool(self.progress_name, error='Ошибка чтения csv-файла')
                        logger.exception('Ошибка чтения csv-файла', extra={'type': "warning"})
                        raise
                    current_col_name = '_'.join(col_name.split('_')[1:])
                    idx = column_names.index(current_col_name)
                    try:
                        task = creation_data.columns_processing[
                            str(creation_data.inputs.get(key).parameters.cols_names[idx][0])].type
                    except IndexError:
                        task = LayerInputTypeChoice.Raw

                    if creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe and \
                            task == LayerInputTypeChoice.Classification:
                        if creation_data.columns_processing[
                            str(creation_data.inputs.get(key).parameters.cols_names[
                                    idx][0])].parameters.one_hot_encoding:
                            encoding = LayerEncodingChoice.ohe
                else:
                    task = creation_data.inputs.get(key).type

                prep = None
                if self.preprocessing.preprocessing.get(key) and \
                        self.preprocessing.preprocessing.get(key).get(col_name):
                    prep = self.preprocessing.preprocessing.get(key).get(col_name)

                if creation_data.inputs.get(key).type == LayerInputTypeChoice.Dataframe:
                    if 'depth' in data.parameters.keys() and data.parameters['depth']:
                        data_to_pass = data.instructions[0:data.parameters['length']]
                    else:
                        data_to_pass = data.instructions[0]
                    c_name = '_'.join(col_name.split('_')[1:])
                    c_idx = column_names.index(c_name)
                    if creation_data.inputs.get(key).parameters.cols_names[c_idx]:
                        c_data_idx = creation_data.inputs.get(key).parameters.cols_names[c_idx][0]
                        if decamelize(creation_data.columns_processing.get(str(c_data_idx)).type) in PATH_TYPE_LIST:
                            data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                elif decamelize(creation_data.inputs.get(key).type) in PATH_TYPE_LIST:
                    data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                else:
                    data_to_pass = data.instructions[0]

                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass, **data.parameters,
                                                                                   **{'preprocess': prep})

                array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                         **arr['parameters'])
                array = array[0] if isinstance(array, tuple) else array
                if not array.shape:
                    array = np.expand_dims(array, 0)
                input_array.append(array)

                classes_names = sorted([os.path.basename(x) for x in
                                        creation_data.inputs.get(key).parameters.sources_paths]) \
                    if not os.path.isfile(creation_data.inputs.get(key).parameters.sources_paths[0]) else \
                    arr['parameters'].get('classes_names')

                num_classes = len(classes_names) if classes_names else None

                # Прописываем параметры для колонки
                current_column = DatasetInputsData(datatype=DataType.get(len(array.shape), 'DIM'),
                                                   dtype=str(array.dtype),
                                                   shape=array.shape,
                                                   name=creation_data.inputs.get(key).name,
                                                   task=task,
                                                   classes_names=classes_names,
                                                   num_classes=num_classes,
                                                   encoding=encoding
                                                   )
                self.columns[key].update([(col_name, current_column.native())])

            # Прописываем параметры для входа
            timeseries_flag = False
            if creation_data.columns_processing:
                for data in creation_data.columns_processing.values():
                    if data.type == ColumnProcessingTypeChoice.Timeseries:
                        timeseries_flag = True
            if not timeseries_flag:
                input_array = np.concatenate(input_array, axis=0)
            else:
                input_array = self.postprocess_timeseries(input_array)
            task, classes_colors, classes_names, encoding, num_classes = None, None, None, None, None
            if len(self.columns[key]) == 1 and not self.columns_processing:
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
                    if len(self.columns[key]) == 1:
                        task = data['task']
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
        for key in self.instructions.outputs.keys():
            self.columns[key] = {}
            output_array = []
            iters = 1
            data = None
            for col_name, data in self.instructions.outputs[key].items():
                prep = None
                if self.preprocessing.preprocessing.get(key) and \
                        self.preprocessing.preprocessing.get(key).get(col_name):
                    prep = self.preprocessing.preprocessing.get(key).get(col_name)
                if decamelize(creation_data.outputs.get(key).type) in PATH_TYPE_LIST or \
                        creation_data.columns_processing.get(str(key)) is not None and \
                        decamelize(creation_data.columns_processing.get(str(key)).type) in PATH_TYPE_LIST:
                    data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                elif 'trend' in data.parameters.keys():
                    if data.parameters['trend']:
                        data_to_pass = [data.instructions[0], data.instructions[data.parameters['length']]]
                    else:
                        data_to_pass = data.instructions[data.parameters['length']:data.parameters['length'] +
                                                                                   data.parameters['depth']]
                elif decamelize(creation_data.outputs.get(key).type) == decamelize(
                        LayerOutputTypeChoice.ObjectDetection):
                    data_to_pass = self.dataframe['train'].iloc[0, 1]
                    tmp_im = self.dataframe['train'].iloc[0, 0].split(';')[1].split(',')
                    data.parameters.update([('orig_x', int(tmp_im[0])),
                                            ('orig_y', int(tmp_im[1]))])
                else:
                    data_to_pass = data.instructions[0]
                # array = np.array([1])
                # if not self.tags[key][col_name] in ['discriminator', 'generator']:
                arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass, **data.parameters,
                                                                                   **{'preprocess': prep})

                array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                         **arr['parameters'])
                if isinstance(array, list):  # Условие для ObjectDetection
                    output_array = [arr for arr in array]
                elif isinstance(array, tuple):
                    array = array[0]
                else:
                    if not array.shape:
                        array = np.expand_dims(array, 0)
                    output_array.append(array)

                cl_names = data.parameters.get('classes_names')
                classes_names = cl_names if cl_names else \
                    sorted([os.path.basename(x) for x in creation_data.outputs.get(key).parameters.sources_paths])
                num_classes = len(classes_names)

                if creation_data.outputs.get(key).type == LayerOutputTypeChoice.Dataframe:
                    try:
                        _, enc = autodetect_encoding(creation_data.outputs.get(key).parameters.sources_paths[0], True)
                        column_names = pd.read_csv(creation_data.outputs.get(key).parameters.sources_paths[0], nrows=0,
                                                   sep=None, engine='python', encoding=enc).columns.to_list()
                    except Exception:
                        progress.pool(self.progress_name, error='Ошибка чтения csv-файла')
                        logger.exception('Ошибка чтения csv-файла', extra={'type': "warning"})
                        raise
                    current_col_name = '_'.join(col_name.split('_')[1:])
                    idx = column_names.index(current_col_name)
                    try:
                        if creation_data.columns_processing[
                            str(creation_data.outputs.get(key).parameters.cols_names[idx][0])].type == \
                                LayerOutputTypeChoice.Timeseries and creation_data.columns_processing[
                            str(creation_data.outputs.get(key).parameters.cols_names[idx][0])].parameters.trend is True:
                            task = LayerOutputTypeChoice.TimeseriesTrend
                        else:
                            task = creation_data.columns_processing[
                                str(creation_data.outputs.get(key).parameters.cols_names[idx][0])].type
                    except IndexError:
                        task = LayerOutputTypeChoice.Raw
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
                elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.Segmentation or\
                        creation_data.outputs.get(key).type == LayerOutputTypeChoice.Dataframe and\
                        creation_data.columns_processing[
                            str(creation_data.outputs.get(key).parameters.cols_names[idx][0])].type\
                        == LayerOutputTypeChoice.Segmentation or\
                        task == LayerOutputTypeChoice.TimeseriesTrend:
                    encoding = LayerEncodingChoice.ohe
                elif creation_data.outputs.get(key).type == LayerOutputTypeChoice.TextSegmentation:
                    encoding = LayerEncodingChoice.multi
                else:
                    encoding = LayerEncodingChoice.none

                if not creation_data.outputs.get(key).type == LayerOutputTypeChoice.ObjectDetection:
                    array = np.expand_dims(array, 0)

                else:
                    iters = 3
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
                    if not creation_data.outputs.get(key).type == LayerOutputTypeChoice.ObjectDetection:
                        self.columns[key].update([(col_name, current_output.native())])
                    else:
                        self.columns[key + i] = {col_name: current_output.native()}

            depth_flag = False
            if not creation_data.outputs.get(key).type == LayerOutputTypeChoice.ObjectDetection:
                if 'depth' in data.parameters.keys() and data.parameters['depth']:
                    depth_flag = True
                    if 'trend' in data.parameters.keys() and data.parameters['trend']:
                        output_array = np.array(output_array[0])
                    else:
                        output_array = self.postprocess_timeseries(output_array)
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
                tmp_tasks = []
                task = LayerInputTypeChoice.Dataframe
                encoding = LayerEncodingChoice.none
                classes_colors, classes_names, = [], []
                for c_name, data in self.columns[key].items():
                    tmp_tasks.append(data['task'])
                    if data['classes_colors']:
                        classes_colors += data['classes_colors']
                    if data['classes_names']:
                        classes_names += data['classes_names']
                if len(set(tmp_tasks)) == 1:
                    task = tmp_tasks[0]
                num_classes = len(classes_names) if classes_names else None
            for i in range(iters):
                if depth_flag:
                    shp = output_array.shape
                else:
                    shp = output_array[i].shape
                current_output = DatasetOutputsData(datatype=DataType.get(len(output_array[i].shape), 'DIM'),
                                                    dtype=str(output_array[i].dtype),
                                                    shape=shp,
                                                    name=creation_data.outputs.get(key).name,
                                                    task=task,
                                                    classes_colors=classes_colors,
                                                    classes_names=classes_names,
                                                    num_classes=num_classes,
                                                    encoding=encoding
                                                    )
                creating_outputs_data[key + i] = current_output.native()

        return creating_outputs_data

    def create_service_parameters(self, creation_data: CreationData) -> dict:

        # Пока сделано только для OD
        creating_service_data = {}
        for key in self.instructions.outputs.keys():
            for col_name, data in self.instructions.outputs[key].items():
                if data.parameters['put_type'] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    prep = None
                    if self.preprocessing.preprocessing.get(key) and self.preprocessing.preprocessing.get(key).get(
                            col_name):
                        prep = self.preprocessing.preprocessing.get(key).get(col_name)

                    if decamelize(creation_data.outputs.get(key).type) in PATH_TYPE_LIST:
                        data_to_pass = os.path.join(self.paths.basepath, data.instructions[0])
                    else:
                        data_to_pass = data.instructions[0]

                    arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(data_to_pass, **data.parameters,
                                                                                       **{'preprocess': prep})

                    array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(arr['instructions'],
                                                                                             **arr['parameters'])
                    classes_names = arr['parameters'].get('classes_names')
                    num_classes = len(classes_names) if classes_names else None
                    for n in range(3):
                        service_data = DatasetOutputsData(datatype=DataType.get(len(array[n + 3].shape), 'DIM'),
                                                          dtype=str(array[n + 3].dtype),
                                                          shape=array[n + 3].shape,
                                                          name=creation_data.outputs.get(key).name,
                                                          task=LayerOutputTypeChoice.ObjectDetection,
                                                          classes_names=classes_names,
                                                          num_classes=num_classes,
                                                          encoding=LayerEncodingChoice.ohe
                                                          )
                        creating_service_data[key + n] = service_data.native()

        return creating_service_data

    def create_dataset_arrays(self, put_data: dict):

        def array_creation(row, instructions):

            full_array = []
            # augm_data = ''
            for h in range(len(row)):
                try:
                    arr = getattr(CreateArray(), f'create_{instructions[h]["put_type"]}')(row[h], **instructions[h])
                    arr = getattr(CreateArray(), f'preprocess_{instructions[h]["put_type"]}')(arr['instructions'],
                                                                                              **arr['parameters'])
                    # if isinstance(arr, tuple):
                    #     full_array.append(arr[0])
                    #     augm_data += arr[1]
                    # else:
                    full_array.append(arr)
                except Exception:
                    progress.pool(self.progress_name, error='Ошибка создания массивов данных')
                    logger.exception('Ошибка создания массивов данных', extra={'type': "warning"})
                    raise

            return full_array  # , augm_data

        for split in ['train', 'val']:
            open_mode = 'w' if not self.paths.arrays.joinpath('dataset.h5').is_file() else 'a'
            hdf = h5py.File(self.paths.arrays.joinpath('dataset.h5'), open_mode)
            if not self.use_generator:
                os.makedirs(self.paths.arrays.joinpath(split), exist_ok=True)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                hdf[split].create_group(f'id_{key}')
                col_name = None
                length, depth, step = 0, 0, 1

                for col_name, data in put_data[key].items():
                    depth = data.parameters['depth'] if 'depth' in data.parameters.keys() and \
                                                        data.parameters['depth'] else 0
                    length = data.parameters['length'] if depth else 0
                    step = data.parameters['step'] if depth else 1

                for j in range(6):
                    globals()[f'current_arrays_{j}'] = []

                data_to_pass = []
                dict_to_pass = []
                for i in range(0, len(self.dataframe[split]) - length - depth, step):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        if not data.parameters['put_type'] in ['noise', 'discriminator', 'generator']:
                            parameters_to_pass = data.parameters.copy()
                            if self.preprocessing.preprocessing.get(key) and \
                                    self.preprocessing.preprocessing.get(key).get(col_name):
                                prep = self.preprocessing.preprocessing.get(key).get(col_name)
                                parameters_to_pass.update([('preprocess', prep)])

                            if self.tags[key][col_name] in PATH_TYPE_LIST:
                                tmp_data.append(os.path.join(self.paths.basepath,
                                                             self.dataframe[split].loc[i, col_name]))
                            elif 'depth' in data.parameters.keys() and data.parameters['depth']:
                                if 'trend' in data.parameters.keys() and data.parameters['trend']:
                                    tmp_data.append([self.dataframe[split].loc[i, col_name],
                                                     self.dataframe[split].loc[i + data.parameters['length'],
                                                                               col_name]])
                                elif 'trend' in data.parameters.keys():
                                    tmp_data.append(
                                        self.dataframe[split].loc[i + data.parameters['length']:i +
                                                                                                data.parameters['length']
                                                                                                + data.parameters[
                                                                                                    'depth'] - 1, col_name])
                                else:
                                    tmp_data.append(self.dataframe[split].loc[i:i + data.parameters['length'] - 1,
                                                    col_name])

                            elif self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                                # if self.augmentation[split]['1_image']:
                                #     tmp_data.append(self.augmentation[split]['1_image'][i])
                                # else:
                                tmp_data.append(self.dataframe[split].loc[i, col_name])
                                tmp_im = self.dataframe['train'].iloc[i, 0].split(';')[1].split(',')
                                parameters_to_pass.update([('orig_x', int(tmp_im[0])),
                                                           ('orig_y', int(tmp_im[1]))])
                            else:
                                tmp_data.append(self.dataframe[split].loc[i, col_name])
                            # if self.tags[key][col_name] == decamelize(LayerInputTypeChoice.Image) and\
                            #         '2_object_detection' in self.dataframe[split].columns:
                            #     parameters_to_pass.update([('augm_data', self.dataframe[split].loc[i, '2_object_detection'])])
                            tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(self.progress_name,
                              message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                              percent=0)
                # if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                #     self.augmentation[split] = {col_name: []}
                if not self.tags[key][col_name] in ['noise', 'discriminator', 'generator']:
                    current_arrays: list = []
                    if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                        for n in range(3):
                            current_group = f'id_{key + n}'
                            current_serv_group = f'id_{key + n}_service'
                            if current_group not in list(hdf[split].keys()):
                                hdf[split].create_group(current_group)
                            if current_serv_group not in list(hdf[split].keys()):
                                hdf[split].create_group(current_serv_group)
                            globals()[f'current_arrays_{n}'] = []
                            globals()[f'current_arrays_{n+3}'] = []
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(array_creation, data_to_pass, dict_to_pass)
                        for i, result in enumerate(results):
                            if psutil.virtual_memory()._asdict().get("percent") > 90:
                                current_arrays = []
                                progress.pool(self.progress_name,
                                              error='Создание датасета прервано. Превышен доступный лимит ОЗУ.')
                                raise Resource
                            # if isinstance(result, tuple):
                            #     augm_data = result[1]
                            #     result = result[0]
                            #     if not augm_data:
                            #         augm_data = ''
                            progress.pool(self.progress_name, percent=ceil(i / len(data_to_pass) * 100))

                            if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                                if depth:
                                    if 'trend' in dict_to_pass[i][0].keys() and dict_to_pass[i][0]['trend']:
                                        array = np.array(result)  # result[0]
                                    else:
                                        array = self.postprocess_timeseries(result)
                                else:
                                    array = np.concatenate(result, axis=0)
                                if self.use_generator:
                                    hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array, compression="gzip")
                                else:
                                    current_arrays.append(array)

                                # if isinstance(augm_data, str):
                                #     self.augmentation[split][col_name].append(augm_data)
                            else:
                                if self.use_generator:
                                    for n in range(3):
                                        hdf[f'{split}/id_{key + n}'].create_dataset(str(i), data=result[0][n], compression="gzip")
                                        hdf[f'{split}/id_{key + n}_service'].create_dataset(str(i), data=result[0][n + 3], compression="gzip")
                                else:
                                    for n in range(6):
                                        globals()[f'current_arrays_{n}'].append(result[0][n])
                            del result

                    if not self.use_generator:
                        if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            for n in range(3):
                                joblib.dump(np.array(globals()[f'current_arrays_{n}']),
                                            os.path.join(self.paths.arrays, split, f'{key + n}.gz'))
                                joblib.dump(np.array(globals()[f'current_arrays_{n + 3}']),
                                            os.path.join(self.paths.arrays, split, f'{key + n}_service.gz'))
                                del globals()[f'current_arrays_{n}']
                                del globals()[f'current_arrays_{n + 3}']
                        else:
                            joblib.dump(np.array(current_arrays), os.path.join(self.paths.arrays, split, f'{key}.gz'))
                            del current_arrays
            hdf.close()

    def write_instructions_to_files(self):

        parameters_path = os.path.join(self.paths.instructions, 'parameters')
        tables_path = os.path.join(self.paths.instructions, 'tables')

        os.makedirs(parameters_path, exist_ok=True)

        for cols in self.instructions.inputs.values():
            for col_name, data in cols.items():
                with open(os.path.join(parameters_path, f'{col_name}.json'), 'w') as cfg:
                    json.dump(data.parameters, cfg)

        for cols in self.instructions.outputs.values():
            for col_name, data in cols.items():
                with open(os.path.join(parameters_path, f'{col_name}.json'), 'w') as cfg:
                    json.dump(data.parameters, cfg)

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

        # Выбор архитектуры
        inp_tasks = []
        out_tasks = []
        for key, val in self.inputs.items():
            if val['task'] == LayerInputTypeChoice.Dataframe:
                tmp = []
                for value in self.columns[key].values():
                    tmp.append(value['task'])
                unique_vals = list(set(tmp))
                if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() and unique_vals[0] \
                        in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
                            LayerInputTypeChoice.Audio, LayerInputTypeChoice.Video]:
                    inp_tasks.append(unique_vals[0])
                else:
                    inp_tasks.append(val['task'])
            else:
                inp_tasks.append(val['task'])

        for key, val in self.outputs.items():
            if val['task'] == LayerOutputTypeChoice.Dataframe:
                tmp = []
                for value in self.columns[key].values():
                    tmp.append(value['task'])
                unique_vals = list(set(tmp))
                if len(unique_vals) == 1 and unique_vals[0] in LayerOutputTypeChoice.__dict__.keys():
                    out_tasks.append(unique_vals[0])
                else:
                    out_tasks.append(val['task'])
            else:
                out_tasks.append(val['task'])

        inp_task_name = list(set(inp_tasks))[0] if len(set(inp_tasks)) == 1 else LayerInputTypeChoice.Dataframe
        out_task_name = list(set(out_tasks))[0] if len(set(out_tasks)) == 1 else LayerOutputTypeChoice.Dataframe

        if inp_task_name + out_task_name in ArchitectureChoice.__dict__.keys():
            architecture = ArchitectureChoice.__dict__[inp_task_name + out_task_name]
        elif out_task_name in ArchitectureChoice.__dict__.keys():
            architecture = ArchitectureChoice.__dict__[out_task_name]
        elif out_task_name == LayerOutputTypeChoice.ObjectDetection:
            architecture = ArchitectureChoice.__dict__[creation_data.outputs.get(2).parameters.model.title() +
                                                       creation_data.outputs.get(2).parameters.yolo.title()]
        elif out_task_name == LayerOutputTypeChoice.Tracker:
            architecture = ArchitectureChoice.Tracker
        elif out_task_name == LayerOutputTypeChoice.Speech2Text:
            architecture = ArchitectureChoice.Speech2Text
        elif out_task_name == LayerOutputTypeChoice.Text2Speech:
            architecture = ArchitectureChoice.Text2Speech
        elif creation_data.outputs[0].type in [LayerOutputTypeChoice.Discriminator, LayerOutputTypeChoice.Generator] \
                and len(creation_data.inputs) == 2:
            architecture = ArchitectureChoice.ImageGAN
        elif creation_data.outputs[0].type in [LayerOutputTypeChoice.Discriminator, LayerOutputTypeChoice.Generator] \
                and len(creation_data.inputs) > 2:
            architecture = ArchitectureChoice.ImageCGAN
        elif inp_task_name == out_task_name == 'Text':
            architecture = ArchitectureChoice.TextTransformer
        # elif creation_data.outputs[0].type in [LayerOutputTypeChoice.Discriminator, LayerOutputTypeChoice.Generator] \
        #         and len(creation_data.inputs) > 2:
        #     architecture = ArchitectureChoice.CGAN
        else:
            architecture = ArchitectureChoice.Basic
        out_list = []
        for key, val in self.outputs.items():
            out_list.append(val['task'])
        if out_list == ['Generator', 'Discriminator']:
            if len(self.inputs) == 2:
                architecture = ArchitectureChoice.ImageGAN
            elif len(self.inputs) == 3:
                architecture = ArchitectureChoice.ImageCGAN

        data = {'name': creation_data.name,
                'alias': creation_data.alias,
                'group': DatasetGroupChoice.custom,
                'use_generator': creation_data.use_generator,
                'tags': tags_list,
                'user_tags': creation_data.tags,
                # 'language': '',  # зачем?...
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'architecture': architecture,
                'size': {'value': size_bytes}
                }

        for attr in ['inputs', 'outputs', 'columns', 'service']:
            if attr in self.__dict__.keys():
                data[attr] = self.__dict__[attr]

        with open(os.path.join(self.paths.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(DatasetData(**data).native(), fp)
        logger.debug(DatasetData(**data).native())

        return data

    @staticmethod
    def zip_dataset(src, dst):
        zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
        abs_src = os.path.abspath(src)
        for dirname, subdirs, files in os.walk(src):
            for filename in files:
                absname = os.path.abspath(os.path.join(dirname, filename))
                arcname = absname[len(abs_src) + 1:]
                zf.write(absname, arcname)
        zf.close()
