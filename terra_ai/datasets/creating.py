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
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG
from terra_ai.data.datasets.creations.layers.output.types.ObjectDetection import LayerODDatasetTypeChoice
from terra_ai import progress

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
from distutils.dir_util import copy_tree
from math import ceil
from PIL import Image
from itertools import repeat
from pathlib import Path
from typing import Union
from datetime import datetime
from pytz import timezone


class CreateDataset(object):

    def __init__(self, cr_data: CreationData):

        creation_data = cr_data  # ВРЕМЕННО!!!
        self.temp_directory: str = tempfile.mkdtemp()
        os.makedirs(Path(self.temp_directory, f'{creation_data.alias}.{DATASET_EXT}_NEW'), exist_ok=True)
        self.dataset_paths_data = DatasetPathsData(
            basepath=Path(self.temp_directory, f'{creation_data.alias}.{DATASET_EXT}_NEW'))
        copy_tree(str(creation_data.source_path), str(self.dataset_paths_data.sources))
        self.version = CreateVersion(version_data=creation_data.version,
                                     dataset_basepath=self.dataset_paths_data.basepath)

        self.zip_dataset(self.dataset_paths_data.sources, os.path.join(self.temp_directory, 'sources'))
        shutil.move(os.path.join(self.temp_directory, 'sources.zip'), self.dataset_paths_data.basepath)
        #         for key, value in self.dataset_paths_data.__dict__.items():
        #             if not key == 'basepath':
        #                 shutil.rmtree(value)
        shutil.rmtree(self.dataset_paths_data.sources)
        self.write_dataset_configure(creation_data)
        #         self.datasetdata = DatasetData(**self.write_dataset_configure(creation_data=creation_data))

        if Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}_NEW').is_dir():
            shutil.rmtree(Path(creation_data.datasets_path, f'{creation_data.alias}.{DATASET_EXT}_NEW'))
        shutil.move(str(self.dataset_paths_data.basepath), creation_data.datasets_path)
        shutil.rmtree(self.temp_directory)

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
                'group': DatasetGroupChoice.custom,
                'tags': tags_list,
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'architecture': creation_data.task_type,
                }

        with open(os.path.join(self.dataset_paths_data.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(DatasetData(**data).native(), fp)


class CreateVersion(object):

    def __init__(self, version_data, dataset_basepath):

        self.y_cls: list = []
        self.tags = {}
        self.dataframe: dict = {}
        self.preprocessing = CreatePreprocessing()

        self.dataset_paths_data = DatasetPathsData(basepath=dataset_basepath)
        print('self.dataset_paths_data:', self.dataset_paths_data)
        if os.listdir(self.dataset_paths_data.versions):
            self.version_id = max([int(x) for x in os.listdir(self.dataset_paths_data.versions)]) + 1
        else:
            self.version_id = 1
        os.makedirs(self.dataset_paths_data.versions.joinpath(str(self.version_id)), exist_ok=True)
        self.version_paths_data = VersionPathsData(
            basepath=self.dataset_paths_data.versions.joinpath(str(self.version_id)))
        print('self.version_paths_data', self.version_paths_data)
        self.instructions: DatasetInstructionsData = self.create_instructions(version_data)
        self.create_preprocessing(self.instructions)
        self.fit_preprocessing(put_data=self.instructions.inputs)
        self.fit_preprocessing(put_data=self.instructions.outputs)
        self.create_table(version_data)

        if not version_data.use_generator:
            x_array = self.create_dataset_arrays(put_data=self.instructions.inputs)
            y_array = self.create_dataset_arrays(put_data=self.instructions.outputs)
            if not isinstance(y_array, dict):
                self.write_arrays(x_array, y_array[0], y_array[1])
            else:
                self.write_arrays(x_array, y_array)

        self.write_instructions_to_files()
        self.zip_dataset(self.version_paths_data.basepath,
                         os.path.join(self.dataset_paths_data.versions, str(self.version_id), 'version'))
        for key, value in self.version_paths_data.__dict__.items():
            if not key == 'basepath':
                shutil.rmtree(value)
        #         shutil.move(os.path.join(self.dataset_paths_data.versions, str(self.version_id), 'version.zip'),
        #                     self.version_paths_data.basepath)
        self.write_version_configure()

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

    def create_instructions(self, version_data):

        inputs = self.create_put_instructions(puts=version_data.inputs, processing=version_data.processing)
        outputs = self.create_put_instructions(puts=version_data.outputs, processing=version_data.processing)

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        return instructions

    def create_put_instructions(self, puts, processing):

        put_parameters = {}

        for idx in range(puts[0].id, puts[0].id + len(puts)):
            data = []
            for path, val in puts.get(idx).parameters.items():
                if path.is_dir():
                    data_to_pass = []
                    for direct, folder, file_name in os.walk(path):
                        if file_name:
                            for name in sorted(file_name):
                                data_to_pass.append(os.path.join(path, name))
                    parameters = processing[str(val[os.path.basename(path)][0])].native()  # Аккуратно с [0]
                    self.tags[idx] = {
                        f'{puts.get(idx).id}_{decamelize(parameters["type"])}': decamelize(parameters['type'])}
                    if parameters['type'] == LayerOutputTypeChoice.Classification:
                        data_to_pass = self.y_cls

                elif path.is_file():
                    print('ТАБЛИЦА')

                instr = getattr(CreateArray, f'instructions_{decamelize(parameters["type"])}')(data_to_pass,
                                                                                               **parameters[
                                                                                                   'parameters'])
                cut = getattr(CreateArray, f'cut_{decamelize(parameters["type"])}')(instr['instructions'],
                                                                                    **instr['parameters'], **{
                        'cols_names': decamelize(parameters["type"]), 'put': idx})
                for i in range(len(cut['instructions'])):
                    if parameters['type'] != LayerOutputTypeChoice.Classification:
                        if decamelize(parameters['type']) in PATH_TYPE_LIST:
                            data.append(os.path.join('sources',
                                                     cut['instructions'][i].replace(str(self.dataset_paths_data.sources), '')[
                                                     1:]))
                        else:
                            data.append(cut['instructions'][i])
                        self.y_cls.append(os.path.basename(path))

            if parameters['type'] != LayerOutputTypeChoice.Classification:
                instructions_data = InstructionsData(instructions=data, parameters=cut['parameters'])
            else:
                instructions_data = InstructionsData(instructions=self.y_cls, parameters=cut['parameters'])
            instructions_data.parameters.update([('put_type', decamelize(parameters['type']))])
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
                # if 'augmentation' in data.parameters.keys() and data.parameters['augmentation']:
                # self.augmentation[data.parameters['cols_names']] = {'train': [], 'val': [], 'test': []}
                # {'object': self.preprocessing.create_image_augmentation(data.parameters['augmentation']),
                # 'data': []}

    def fit_preprocessing(self, put_data):

        for key in put_data.keys():
            for col_name, data in put_data[key].items():
                if 'scaler' in data.parameters and data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler,
                                                                                     None]:
                    #                     progress.pool(self.progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
                    #                     try:
                    if self.tags[key][col_name] in PATH_TYPE_LIST:
                        for i in range(len(data.instructions)):
                            #                             progress.pool(self.progress_name,
                            #                                           percent=ceil(i / len(data.instructions) * 100))

                            arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
                                os.path.join(self.dataset_paths_data.basepath, data.instructions[i]),
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

    #                     except Exception:
    #                         progress.pool(self.progress_name, error='Ошибка обучения скейлера')
    #                         raise

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
        #         try:
        dataframe = pd.DataFrame(build_dataframe)
        #         except Exception:
        #             progress.pool(self.progress_name,
        #                           error='Ошибка создания датасета. Нессответствие количества входных/выходных данных')
        #             raise
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)
        # print(self.dataframe['train'])

    def create_dataset_arrays(self, put_data: dict):

        def array_creation(row, instructions):

            full_array = []
            augm_data = ''
            for h in range(len(row)):
                try:
                    arr = getattr(CreateArray(), f'create_{instructions[h]["put_type"]}')(row[h], **instructions[h])
                    arr = getattr(CreateArray(), f'preprocess_{instructions[h]["put_type"]}')(arr['instructions'],
                                                                                              **arr['parameters'])
                    if isinstance(arr, tuple):
                        full_array.append(arr[0])
                        augm_data += arr[1]
                    else:
                        full_array.append(arr)
                except Exception:
                    #                     progress.pool(self.progress_name, error='Ошибка создания массивов данных')
                    raise

            return full_array, augm_data

        out_array = {'train': {}, 'val': {}}
        service = {'train': {}, 'val': {}}

        for split in list(out_array.keys()):
            for key in put_data.keys():
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
                        parameters_to_pass = data.parameters.copy()
                        if self.preprocessing.preprocessing.get(key) and \
                                self.preprocessing.preprocessing.get(key).get(col_name):
                            prep = self.preprocessing.preprocessing.get(key).get(col_name)
                            parameters_to_pass.update([('preprocess', prep)])

                        if self.tags[key][col_name] in PATH_TYPE_LIST:
                            tmp_data.append(os.path.join(self.dataset_paths_data.basepath,
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
                            #                             if self.augmentation[split]['1_image']:
                            #                                 tmp_data.append(self.augmentation[split]['1_image'][i])
                            #                             else:
                            tmp_data.append(self.dataframe[split].loc[i, col_name])
                            tmp_im = Image.open(os.path.join(self.dataset_paths_data.basepath,
                                                             self.dataframe[split].iloc[i, 0]))
                            parameters_to_pass.update([('orig_x', tmp_im.width),
                                                       ('orig_y', tmp_im.height)])
                        else:
                            tmp_data.append(self.dataframe[split].loc[i, col_name])
                        if self.tags[key][col_name] == decamelize(LayerInputTypeChoice.Image) and \
                                '2_object_detection' in self.dataframe[split].columns:
                            parameters_to_pass.update(
                                [('augm_data', self.dataframe[split].loc[i, '2_object_detection'])])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                #                 progress.pool(self.progress_name,
                #                               message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                #                               percent=0)
                #                 if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                #                     self.augmentation[split] = {col_name: []}
                current_arrays: list = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(array_creation, data_to_pass, dict_to_pass)
                    for i, result in enumerate(results):
                        #                         if psutil.virtual_memory()._asdict().get("percent") > 90:
                        #                             current_arrays = []
                        #                             raise Resource
                        if isinstance(result, tuple):
                            augm_data = result[1]
                            result = result[0]
                            if not augm_data:
                                augm_data = ''
                        #                         progress.pool(self.progress_name, percent=ceil(i / len(data_to_pass) * 100))
                        if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            if depth:
                                if 'trend' in dict_to_pass[i][0].keys() and dict_to_pass[i][0]['trend']:
                                    array = np.array(result[0])
                                else:
                                    array = self.postprocess_timeseries(result)
                            else:
                                array = np.concatenate(result, axis=0)
                            current_arrays.append(array)
                        #                             if isinstance(augm_data, str):
                        #                                 self.augmentation[split][col_name].append(augm_data)
                        else:
                            for n in range(6):
                                globals()[f'current_arrays_{n}'].append(result[0][n])

                if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    for n in range(3):
                        out_array[split][key + n] = np.array(globals()[f'current_arrays_{n}'])
                        service[split][key + n] = np.array(globals()[f'current_arrays_{n + 3}'])
                        # print(np.array(globals()[f'current_arrays_{n}']).shape)
                        # print(np.array(globals()[f'current_arrays_{n + 3}']).shape)
                else:
                    out_array[split][key] = np.array(current_arrays)
                    # print(out_array[split][key].shape)

        if service['train']:
            return out_array, service
        else:
            return out_array

    def write_arrays(self, array_x, array_y, array_service=None):

        for array in [array_x, array_y]:
            for sample in array.keys():
                for put in array[sample].keys():
                    os.makedirs(os.path.join(self.version_paths_data.arrays, sample), exist_ok=True)
                    joblib.dump(array[sample][put], os.path.join(self.version_paths_data.arrays, sample, f'{put}.gz'))
        if array_service:
            for sample in array_service.keys():
                for put in array_service[sample].keys():
                    joblib.dump(array_service[sample][put],
                                os.path.join(self.version_paths_data.arrays, sample, f'{put}_service.gz'))

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

        pass

    def write_version_configure(self):
        """
        inputs, outputs, service, size, use_generator, columns, date
        """

        size_bytes = 0
        for path, dirs, files in os.walk(self.version_paths_data.basepath):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        data = {'name': f"Вариант {self.version_id}",
                'alias': f"variant_{self.version_id}",
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'size': {'value': size_bytes}
                }

        with open(os.path.join(self.version_paths_data.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(VersionData(**data).native(), fp)
