from terra_ai.data.datasets.creations.layers.image_augmentation import AugmentationData
from terra_ai.utils import decamelize, camelize
from terra_ai.exceptions.tensor_flow import ResourceExhaustedError as Resource
from terra_ai.datasets.data import DataType, InstructionsData, DatasetInstructionsData
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.data.datasets.creation import CreationData, CreationInputsList, CreationOutputsList, CreationVersionData
from terra_ai.data.datasets.dataset import DatasetData, DatasetInputsData, DatasetOutputsData, DatasetPathsData, \
    VersionPathsData, VersionData
from terra_ai.data.datasets.extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerPrepareMethodChoice, LayerScalerImageChoice, ColumnProcessingTypeChoice, \
    LayerTypeProcessingClassificationChoice, LayerEncodingChoice
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, VERSION_EXT, VERSION_CONFIG
from terra_ai.data.datasets.creations.layers.output.types.ObjectDetection import LayerODDatasetTypeChoice
from terra_ai import progress

import h5py
import psutil
import cv2
import os
import random
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
import zipfile
import concurrent.futures
from distutils.dir_util import copy_tree
from math import ceil
from PIL import Image
from itertools import repeat
from pathlib import Path
from datetime import datetime
from pytz import timezone
from terra_ai.logging import logger


class CreateDataset(object):

    progress_name: str = 'create_dataset'

    @progress.threading
    def __init__(self, creation_data: CreationData):

        progress.pool.reset(name=self.progress_name,
                            message='Начало',
                            finished=False)
        logger.info(f'Начало формирования датасета {creation_data.name}.')
        self.temp_directory: Path = Path(tempfile.mkdtemp())
        os.makedirs(self.temp_directory.joinpath('.'.join([creation_data.alias, DATASET_EXT])), exist_ok=True)
        self.dataset_paths_data: DatasetPathsData = DatasetPathsData(
            basepath=self.temp_directory.joinpath('.'.join([creation_data.alias, DATASET_EXT])))
        progress.pool(name=self.progress_name, message='Копирование файлов', percent=10)
        copy_tree(str(creation_data.source_path), str(self.dataset_paths_data.sources))
        self.zip_dataset(self.dataset_paths_data.sources, self.temp_directory.joinpath('sources'))
        shutil.move(str(self.temp_directory.joinpath('sources.zip')), self.dataset_paths_data.basepath)
        shutil.rmtree(self.dataset_paths_data.sources)
        dataset_data = self.write_dataset_configure(creation_data)
        if creation_data.datasets_path.joinpath('.'.join([creation_data.alias, DATASET_EXT])).is_dir():
            progress.pool(name=self.progress_name,
                          message=f"Удаление существующего датасета "
                                  f"{creation_data.datasets_path.joinpath('.'.join([creation_data.alias, DATASET_EXT]))}",
                          percent=70)
            shutil.rmtree(creation_data.datasets_path.joinpath('.'.join([creation_data.alias, DATASET_EXT])))
        progress.pool(name=self.progress_name, message=f"Копирование датасета в {creation_data.datasets_path}",
                      percent=80)
        shutil.move(str(self.dataset_paths_data.basepath), creation_data.datasets_path)
        progress.pool(name=self.progress_name, message=f"Удаление временной папки {self.temp_directory}", percent=95)
        shutil.rmtree(self.temp_directory)
        progress.pool(name=self.progress_name, message='Формирование датасета завершено', data=dataset_data,
                      percent=100, finished=True)
        logger.info(f'Создан датасет {creation_data.name}.')

        # if creation_data.version:  # Больше сделано для дебаггинга
        #     self.version = CreateVersion(version_data=creation_data.version)

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

    def write_dataset_configure(self, creation_data):

        tags_list = [{'alias': x, 'name': x.capitalize()} for x in decamelize(creation_data.task_type).split('_')]
        for tag in creation_data.tags:
            tags_list.append(tag.native())

        data = {'name': creation_data.name,
                'alias': creation_data.alias,
                'group': DatasetGroupChoice.trds,
                'tags': tags_list,
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'architecture': creation_data.task_type,
                }
        dataset_data = DatasetData(**data)
        with open(os.path.join(self.dataset_paths_data.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(dataset_data.native(), fp)

        return dataset_data


class CreateVersion(object):

    progress_name: str = 'create_version'

    @progress.threading
    def __init__(self, version_data: CreationVersionData):

        progress.pool.reset(name=self.progress_name, message='Начало', finished=False)

        self.y_cls: list = []
        self.tags = {}
        self.dataframe: dict = {}
        self.columns: dict = {}
        self.preprocessing = CreatePreprocessing()
        version_data = self.preprocess_version_data(version_data)

        logger.info(f'Начало создания версии {version_data.name}.')
        logger.debug(version_data)

        self.temp_directory: Path = Path(tempfile.mkdtemp())
        self.sources_temp_directory: Path = Path(tempfile.mkdtemp())
        self.dataset_paths_data = DatasetPathsData(basepath=self.temp_directory)
        self.parent_dataset_paths_data = DatasetPathsData(
            basepath=version_data.datasets_path.joinpath('.'.join([version_data.parent_alias, DATASET_EXT]))
        )
        progress.pool(name=self.progress_name, message='Копирование исходного архива', percent=0)
        shutil.copyfile(
            self.parent_dataset_paths_data.basepath.joinpath('sources.zip'),
            self.dataset_paths_data.basepath.joinpath('sources.zip')
        )
        current_version = self.dataset_paths_data.versions.joinpath(f'{version_data.alias}.{VERSION_EXT}')
        os.makedirs(current_version)
        self.version_paths_data = VersionPathsData(basepath=current_version)
        progress.pool(name=self.progress_name, message='Распаковка исходного архива', percent=0)
        with zipfile.ZipFile(self.dataset_paths_data.basepath.joinpath('sources.zip'), 'r') as z_file:
            z_file.extractall(self.sources_temp_directory)
        progress.pool(name=self.progress_name, message='Создание инструкций', percent=0)
        self.instructions: DatasetInstructionsData = self.create_instructions(version_data)
        progress.pool(name=self.progress_name, message='Создание объектов обработки', percent=0)
        self.create_preprocessing(self.instructions)
        self.fit_preprocessing(put_data=self.instructions.inputs)
        self.fit_preprocessing(put_data=self.instructions.outputs)
        self.create_table(version_data)
        progress.pool(name=self.progress_name, message='Создание массивов данных', percent=0)
        self.create_dataset_arrays(put_data=self.instructions.inputs)
        self.create_dataset_arrays(put_data=self.instructions.outputs)

        self.inputs = self.create_put_parameters(self.instructions.inputs, version_data, 'inputs')
        self.outputs = self.create_put_parameters(self.instructions.outputs, version_data, 'outputs')
        # self.service = self.create_put_parameters(self.instructions.service, version_data, 'service')

        progress.pool(name=self.progress_name, message='Сохранение', percent=100)
        self.write_instructions_to_files()
        self.zip_dataset(self.version_paths_data.basepath, os.path.join(self.dataset_paths_data.versions, 'version'))
        version_dir = self.parent_dataset_paths_data.versions.joinpath('.'.join([version_data.alias, VERSION_EXT]))
        shutil.rmtree(version_dir) if version_dir.is_dir() else os.makedirs(version_dir)
        shutil.move(self.dataset_paths_data.versions.joinpath('version.zip'), version_dir.joinpath('version.zip'))
        self.write_version_configure(version_data)
        shutil.rmtree(self.sources_temp_directory)
        shutil.rmtree(self.temp_directory)
        progress.pool(name=self.progress_name, message='Формирование версии датасета завершено', data=version_data,
                      percent=100, finished=True)
        logger.info(f'Создана версия {version_data.name}', extra={'type': "info"})

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

    @staticmethod
    def preprocess_version_data(version_data):

        for worker_name, worker_params in version_data.processing.items():
            if version_data.processing[worker_name].type == LayerOutputTypeChoice.Segmentation:
                for w_name in version_data.processing:
                    if version_data.processing[w_name].type == LayerInputTypeChoice.Image:
                        version_data.processing[worker_name].parameters.height =\
                            version_data.processing[w_name].parameters.height
                        version_data.processing[worker_name].parameters.width = \
                            version_data.processing[w_name].parameters.width
            elif version_data.processing[worker_name].type == LayerOutputTypeChoice.TextSegmentation:
                for w_name in version_data.processing:
                    if version_data.processing[w_name].type == LayerOutputTypeChoice.Text:
                        version_data.processing[worker_name].parameters.text_mode = \
                            version_data.processing[w_name].parameters.text_mode
                        version_data.processing[worker_name].parameters.length = \
                            version_data.processing[w_name].parameters.length
                        version_data.processing[worker_name].parameters.step = \
                            version_data.processing[w_name].parameters.step
                        version_data.processing[worker_name].parameters.max_words = \
                            version_data.processing[w_name].parameters.max_words
                        filters = version_data.processing[w_name].parameters.filters
                        for x in version_data.processing[worker_name].parameters.open_tags + version_data.processing[worker_name].parameters.close_tags:
                            filters = filters.replace(x, '')
                        version_data.processing[w_name].parameters.filters = filters
                        version_data.processing[worker_name].parameters.filters = filters
                        version_data.processing[w_name].parameters.open_tags = \
                            version_data.processing[worker_name].parameters.open_tags
                        version_data.processing[w_name].parameters.close_tags = \
                            version_data.processing[worker_name].parameters.close_tags
            elif version_data.processing[worker_name].type == LayerOutputTypeChoice.ObjectDetection:
                for w_name, w_params in version_data.processing.items():
                    if version_data.processing[w_name].type == LayerInputTypeChoice.Image:
                        version_data.processing[worker_name].parameters.frame_mode = \
                            version_data.processing[w_name].parameters.image_mode
                names_list = get_od_names(version_data)
                version_data.processing[worker_name].parameters.classes_names = names_list
                version_data.processing[worker_name].parameters.num_classes = len(names_list)

        return version_data

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

    def create_instructions(self, version_data):

        inputs = self.create_put_instructions(puts=version_data.inputs, processing=version_data.processing)
        outputs = self.create_put_instructions(puts=version_data.outputs, processing=version_data.processing)

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_put_instructions(self, puts, processing):

        def instructions(one_path, params):

            try:
                instr = getattr(CreateArray, f'instructions_{decamelize(params["type"])}')([one_path],
                                                                                           **params['parameters'])
                cut = getattr(CreateArray, f'cut_{decamelize(params["type"])}')(instr['instructions'],
                                                                                self.version_paths_data.sources,
                                                                                **instr['parameters'],
                                                                                **{'cols_names': cols_names,
                                                                                   'put': idx})
            except Exception:
                progress.pool(self.progress_name, error=f'Ошибка создания инструкций для {puts}')
                logger.debug(f'Создание инструкций провалилось на {one_path}')
                raise

            # return_data = []
            # for i in range(len(cut['instructions'])):
            #     if decamelize(parameters['type']) in PATH_TYPE_LIST:
            #         return_data.append(cut['instructions'][i].replace(str(self.version_paths_data.sources), '')[1:])  # self.sources_temp_directory
            #     else:
            #         return_data.append(cut['instructions'][i])
            if idx == puts[0].id and parameters['type'] != LayerOutputTypeChoice.Classification:
                self.y_cls += [os.path.basename(path) for _ in range(len(cut['instructions']))]

            return cut  # {'instructions': return_data, 'parameters': cut['parameters']}

        put_parameters = {}

        for idx in range(puts[0].id, puts[0].id + len(puts)):
            data = []
            parameters = None
            for path, val in puts.get(idx).parameters.items():
                data_to_pass = []
                cols_names = ''
                current_path = self.sources_temp_directory.joinpath(path)
                parameters = processing[str(val[os.path.basename(path)][0])].native()  # Аккуратно с [0]
                if current_path.is_dir():
                    if parameters['type'] == LayerOutputTypeChoice.Classification:
                        data_to_pass = self.y_cls
                    else:
                        for direct, folder, file_name in os.walk(current_path):
                            if file_name:
                                for name in sorted(file_name):
                                    data_to_pass.append(os.path.join(current_path, name))
                    cols_names = f"{puts.get(idx).id}_{decamelize(parameters['type'])}"
                    self.tags[idx] = {cols_names: decamelize(parameters['type'])}
                elif current_path.is_file():
                    print('ТАБЛИЦА')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(instructions, data_to_pass, repeat(parameters))
                    progress.pool(self.progress_name, message=f'Формирование файлов для {os.path.basename(path)}')
                    for i, result in enumerate(results):
                        progress.pool(self.progress_name, percent=ceil(i / len(data_to_pass) * 100))
                        if decamelize(parameters['type']) in PATH_TYPE_LIST:
                            for j in range(len(result['instructions'])):
                                result['instructions'][j] = result['instructions'][j].replace(str(self.version_paths_data.sources), '')[1:]
                        data += result['instructions']
                        result_params = result['parameters']
                        # classes_names += result['parameters']['classes_names']
                        # if idx == puts[0].id and parameters['type'] != LayerOutputTypeChoice.Classification:
                        #     self.y_cls += [os.path.basename(path) for _ in range(len(result['instructions']))]
                # if decamelize(parameters['type']) in PATH_TYPE_LIST:
                #     for i in range(len(data)):
                #         data[i] = data[i].replace(str(self.version_paths_data.sources), '')[1:]  # data[i].replace(str(self.version_paths_data.sources), '')[1:])
                        # data.append(cut['instructions'][i].replace(str(self.version_paths_data.sources), '')[1:]) # self.sources_temp_directory
                    # else:
                    #     data.append(cut['instructions'][i])
                #     # if idx - 1 > 0 and parameters['type'] != prev_type:
                #     # if idx == 0:
                #     # if parameters['type'] != LayerOutputTypeChoice.Classification and i == 0:
                #     #     self.y_cls.append(os.path.basename(path))
                # if idx == puts[0].id and parameters['type'] != LayerOutputTypeChoice.Classification:
                #     self.y_cls += [os.path.basename(path) for _ in range(len(data))]
            if parameters['type'] == LayerOutputTypeChoice.Classification:
                data = self.y_cls
                # ### Дальше идет не очень хороший код
                if parameters['parameters']['type_processing'] == "categorical":
                    classes_names = list(dict.fromkeys(data))
                else:
                    if len(parameters['parameters']["ranges"].split(" ")) == 1:
                        border = max(data) / int(parameters['parameters']["ranges"])
                        classes_names = np.linspace(border, max(data), int(parameters['parameters']["ranges"])).tolist()
                    else:
                        classes_names = parameters['parameters']["ranges"].split(" ")
                result['parameters']['classes_names'] = classes_names
                result['parameters']['num_classes'] = len(classes_names)
                # ###
            data = self.y_cls if parameters['type'] == LayerOutputTypeChoice.Classification else data
            instructions_data = InstructionsData(instructions=data, parameters=result_params)
            instructions_data.parameters.update({'put_type': decamelize(parameters['type'])})
            put_parameters[idx] = {f'{idx}_{decamelize(parameters["type"])}': instructions_data}

        return put_parameters

    def create_preprocessing(self, instructions: DatasetInstructionsData):

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
                    if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                             LayerPrepareMethodChoice.bag_of_words]:
                        self.preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
                    elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                        self.preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)
                else:
                    self.preprocessing.preprocessing.update(
                        {data.parameters['put']: {data.parameters['cols_names']: None}}
                    )

    def fit_preprocessing(self, put_data):

        for key in put_data.keys():
            for col_name, data in put_data[key].items():
                if 'scaler' in data.parameters and data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler,
                                                                                     None]:
                    progress.pool(self.progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
                    #                     try:
                    if self.tags[key][col_name] in PATH_TYPE_LIST:
                        for i in range(len(data.instructions)):
                            #                             progress.pool(self.progress_name,
                            #                                           percent=ceil(i / len(data.instructions) * 100))

                            arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                                self.sources_temp_directory.joinpath(data.instructions[i]),
                                **data.parameters
                            )

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

                        # except Exception:
                        #     progress.pool(self.progress_name, error='Ошибка обучения скейлера')
                        #     raise

    def create_table(self, version_data: CreationVersionData):

        classes_dict = {}
        for out in self.instructions.outputs.keys():
            #             if creation_data.columns_processing.get(str(out)) is not None and \
            #                     creation_data.columns_processing.get(str(out)).type == LayerOutputTypeChoice.Classification and \
            #                     creation_data.columns_processing.get(str(out)).parameters.type_processing != \
            #                     LayerTypeProcessingClassificationChoice.ranges or \
            #                     creation_data.outputs.get(out).type == LayerOutputTypeChoice.Classification:
            #                 for col_name, data in self.instructions.outputs[out].items():
            #                     class_names = list(dict.fromkeys(data.instructions))
            #                     classes_dict = {cl_name: [] for cl_name in class_names}
            #                     for idx, cl_name in enumerate(data.instructions):
            #                         classes_dict[cl_name].append(idx)
            #                 break
            #             else:
            for col_name, data in self.instructions.outputs[out].items():
                classes_dict = {'one_class': [idx for idx in range(len(data.instructions))]}

        if version_data.info.shuffle:
            for key in classes_dict.keys():
                random.shuffle(classes_dict[key])

        split_sequence = {"train": [], "val": []}
        for key, value in classes_dict.items():
            train_len = int(version_data.info.part.train * len(classes_dict[key]))
            split_sequence['train'].extend(value[:train_len])
            split_sequence['val'].extend(value[train_len:])

        if version_data.info.shuffle:
            random.shuffle(split_sequence['train'])
            random.shuffle(split_sequence['val'])

        build_dataframe = {}
        for inp in self.instructions.inputs.keys():
            for key, value in self.instructions.inputs[inp].items():
                build_dataframe[key] = value.instructions
        for out in self.instructions.outputs.keys():
            for key, value in self.instructions.outputs[out].items():
                build_dataframe[key] = value.instructions
        try:
            dataframe = pd.DataFrame(build_dataframe)
        except Exception:
            progress.pool(self.progress_name,
                          error='Ошибка создания датасета. Нессответствие количества входных/выходных данных')
            for key, value in build_dataframe.items():
                logger.debug(key, len(value))
            raise
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)
        print(self.dataframe['train'])

    def create_put_parameters(self, put_instructions, version_data: CreationVersionData, put: str):  # -> dict:

        creating_puts_data = {}
        for key in put_instructions.keys():
            put_array = []
            self.columns[key] = {}
            for col_name, data in put_instructions[key].items():
                data_to_pass = data.instructions[0]
                if self.tags[key][col_name] in PATH_TYPE_LIST:
                    data_to_pass = str(self.version_paths_data.sources.joinpath(data_to_pass))
                create = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                    data_to_pass,
                    **data.parameters,
                    **{'preprocess': self.preprocessing.preprocessing[key][col_name]}
                )
                array = getattr(CreateArray(), f'preprocess_{self.tags[key][col_name]}')(create['instructions'],
                                                                                         **create['parameters'])

                # array = array[0] if isinstance(array, tuple) else array
                # if not array.shape:
                #     array = np.expand_dims(array, 0)
                put_array.append(array)
                if create['parameters'].get('classes_names'):
                    classes_names = create['parameters'].get('classes_names')
                else:
                    classes_names = sorted([os.path.basename(x) for x in version_data.__dict__[put].get(key).parameters.keys()])

                # Прописываем параметры для колонки
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'shape': array.shape,
                                  'name': version_data.__dict__[put].get(key).name,
                                  'task': camelize(data.parameters.get('put_type')),
                                  'classes_names': classes_names,
                                  'classes_colors': data.parameters.get('classes_colors'),
                                  'num_classes': len(classes_names) if classes_names else 0,
                                  'encoding': 'none' if not data.parameters.get('encoding') else data.parameters.get('encoding')}
                current_column = DatasetInputsData(**col_parameters) if put == 'inputs' else DatasetOutputsData(**col_parameters)
                self.columns[key].update([(col_name, current_column.native())])

            put_array = np.concatenate(put_array, axis=0)
            classes_colors_list, classes_names_list, encoding_list, task_list = [], [], [], []
            for value in self.columns[key].values():
                if value.get('classes_colors'):
                    for c_color in value.get('classes_colors'):
                        classes_colors_list.append(c_color)
                if value.get('classes_names'):
                    for c_name in value.get('classes_names'):
                        classes_names_list.append(c_name)
                encoding_list.append(value.get('encoding') if value.get('encoding') else 'none')
                task_list.append(value.get('task'))
            put_parameters = {'datatype': DataType.get(len(put_array.shape), 'DIM'),
                              'dtype': str(put_array.dtype),
                              'shape': put_array.shape,
                              'name': version_data.__dict__[put].get(key).name,
                              'task': task_list[0] if len(task_list) == 1 else 'Dataframe',
                              'classes_names': classes_names_list if classes_names_list else None,
                              'classes_colors': classes_colors_list if classes_colors_list else None,
                              'num_classes': len(classes_names_list) if classes_names_list else None,
                              'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}

            creating_puts_data[key] = DatasetInputsData(**put_parameters).native() if put == 'inputs'\
                else DatasetOutputsData(**put_parameters).native()

        return creating_puts_data

    def create_dataset_arrays(self, put_data: dict):

        def array_creation(row, instructions):

            full_array = []
            for h in range(len(row)):
                try:
                    arr = getattr(CreateArray(), f'create_{instructions[h]["put_type"]}')(row[h], **instructions[h])
                    arr = getattr(CreateArray(), f'preprocess_{instructions[h]["put_type"]}')(arr['instructions'],
                                                                                              **arr['parameters'])
                    full_array.append(arr)
                except Exception:
                    progress.pool(self.progress_name, error='Ошибка создания массивов данных')
                    raise

            return full_array

        for split in ['train', 'val']:
            open_mode = 'w' if not self.version_paths_data.arrays.joinpath('dataset.h5') else 'a'
            hdf = h5py.File(self.version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                col_name = None
                length, depth, step = 0, 0, 1

                for col_name, data in put_data[key].items():
                    depth = data.parameters['depth'] if 'depth' in data.parameters.keys() and \
                                                        data.parameters['depth'] else 0
                    length = data.parameters['length'] if depth else 0
                    step = data.parameters['step'] if depth else 1

                data_to_pass = []
                dict_to_pass = []
                for i in range(0, len(self.dataframe[split]) - length - depth, step):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        if self.preprocessing.preprocessing.get(key) and \
                                self.preprocessing.preprocessing.get(key).get(col_name):
                            prep = self.preprocessing.preprocessing.get(key).get(col_name)
                            parameters_to_pass.update([('preprocess', prep)])

                        if self.tags[key][col_name] in PATH_TYPE_LIST:
                            tmp_data.append(os.path.join(self.version_paths_data.sources, # .self.sources_temp_directory
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
                            tmp_data.append(self.dataframe[split].loc[i, col_name])
                            tmp_im = Image.open(os.path.join(self.sources_temp_directory,
                                                             self.dataframe[split].iloc[i, 0]))
                            parameters_to_pass.update([('orig_x', tmp_im.width),
                                                       ('orig_y', tmp_im.height)])
                        else:
                            tmp_data.append(self.dataframe[split].loc[i, col_name])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(self.progress_name,
                              message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                              percent=0)

                if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    for n in range(3):
                        current_group = f'id_{key + n}'
                        current_serv_group = f'id_{key + n}_service'
                        if current_group not in list(hdf[split].keys()):
                            hdf[split].create_group(current_group)
                        if current_serv_group not in list(hdf[split].keys()):
                            hdf[split].create_group(current_serv_group)
                else:
                    hdf[split].create_group(f'id_{key}')

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(array_creation, data_to_pass, dict_to_pass)
                    for i, result in enumerate(results):
                        progress.pool(self.progress_name, percent=ceil(i / len(data_to_pass) * 100))
                        if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            if depth:
                                if 'trend' in dict_to_pass[i][0].keys() and dict_to_pass[i][0]['trend']:
                                    array = np.array(result[0])
                                else:
                                    array = self.postprocess_timeseries(result)
                            else:
                                array = np.concatenate(result, axis=0)
                            hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array)
                        else:
                            for n in range(3):
                                hdf[f'{split}/id_{key + n}'].create_dataset(str(i), data=result[0][n])
                                hdf[f'{split}/id_{key + n}_service'].create_dataset(str(i), data=result[0][n + 3])
                        del result
            hdf.close()

    def write_instructions_to_files(self):

        parameters_path = os.path.join(self.version_paths_data.instructions, 'parameters')
        tables_path = os.path.join(self.version_paths_data.instructions, 'tables')

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
            self.dataframe[key].to_csv(os.path.join(self.version_paths_data.instructions, 'tables', f'{key}.csv'))

    def write_version_configure(self, version_data):
        """
        inputs, outputs, service, size, use_generator, columns, date
        """

        size_bytes = 0
        for path, dirs, files in os.walk(self.version_paths_data.basepath):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        data = {'alias': version_data.alias,
                'name': version_data.name,
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'size': {'value': size_bytes},
                'use_generator': version_data.use_generator,
                'inputs': self.inputs,
                'outputs': self.outputs,
                # 'service': self.service,
                'columns': self.columns
                }

        with open(self.parent_dataset_paths_data.versions.joinpath(f'{version_data.alias}.{VERSION_EXT}')
                      .joinpath(VERSION_CONFIG), 'w') as fp:
            json.dump(VersionData(**data).native(), fp)
