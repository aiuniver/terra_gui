import concurrent.futures
import os

import h5py
import numpy as np
import pandas as pd
from itertools import repeat
from math import ceil
from pathlib import Path

from terra_ai import progress
from terra_ai.data.datasets.creation import CreationVersionData
from terra_ai.data.datasets.dataset import VersionPathsData, DatasetOutputsData, DatasetInputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerEncodingChoice
from terra_ai.datasets import arrays_classes
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.creating import version_progress_name
from terra_ai.datasets.data import InstructionsData, DataType, DatasetInstructionsData
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.logging import logger
from terra_ai.utils import decamelize, camelize, autodetect_encoding


class BaseClass(object):

    @staticmethod
    def multithreading_instructions(one_path, params, dataset_folder, col_name, id):

        try:
            instruction = getattr(getattr(arrays_classes, decamelize(params["type"])),
                                  f'{params["type"]}Array')().prepare(
                sources=[one_path],
                dataset_folder=dataset_folder,
                **params['parameters'],
                **{'cols_names': col_name, 'put': id}
            )

        except Exception:
            progress.pool(version_progress_name, error=f'Ошибка создания инструкций для {col_name}')
            logger.debug(f'Создание инструкций провалилось на {one_path}')
            raise

        return instruction

    @staticmethod
    def multithreading_array(row, instructions):

        full_array = []
        for h in range(len(row)):
            try:
                create = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
                                 f'{camelize(instructions[h]["put_type"])}Array')().create(
                    source=row[h], **instructions[h])
                prepr = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
                                f'{camelize(instructions[h]["put_type"])}Array')().preprocess(
                    create['instructions'], **create['parameters'])
                full_array.append(prepr)
            except Exception:
                progress.pool(version_progress_name, error='Ошибка создания массивов данных')
                raise

        return full_array

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        return version_data

    @staticmethod
    def collect_data_to_pass(put_data, processing, sources_temp_directory):

        data_to_pass = {}
        parameters_to_pass = {}

        for idx in range(put_data[0].id, put_data[0].id + len(put_data)):
            data_to_pass[idx] = {}
            parameters_to_pass[idx] = {}
            for path, val in put_data.get(idx).parameters.items():
                for name, proc in val.items():
                    collected_data = []
                    parameters = processing[str(proc[0])].native()  # Аккуратно с [0]
                    if Path(sources_temp_directory).joinpath(name.split(':')[0]).is_dir():
                        for folder_name in name.split(':'):
                            current_path = Path(sources_temp_directory).joinpath(folder_name)
                            for direct, folder, files_name in os.walk(current_path):
                                if files_name:
                                    for file_name in sorted(files_name):
                                        collected_data.append(os.path.join(current_path, file_name))
                    elif Path(sources_temp_directory).joinpath(path.split(':')[0]).is_file():
                        current_path = Path(sources_temp_directory).joinpath(path.split(':')[0])
                        _, enc = autodetect_encoding(str(current_path), True)
                        collected_data = pd.read_csv(current_path, sep=None, usecols=[name],
                                                     engine='python', encoding=enc)[name].to_list()
                        if decamelize(parameters['type']) in PATH_TYPE_LIST:
                            collected_data = [str(Path(sources_temp_directory).joinpath(Path(x)))
                                              for x in collected_data]
                    data_to_pass[idx].update({name: collected_data})
                    parameters_to_pass[idx].update({name: parameters})

        return data_to_pass, parameters_to_pass

    def create_put_instructions(self, dictio, parameters, version_sources_path):

        put_instructions = {}
        tags = {}

        for put_id, pass_data_parameters in dictio.items():
            put_instructions[put_id] = {}
            tags[put_id] = {}
            for col_name, data in pass_data_parameters.items():
                instructions = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(self.multithreading_instructions,
                                           data,
                                           repeat(parameters[put_id][col_name]),
                                           repeat(version_sources_path),
                                           repeat(col_name),
                                           repeat(put_id))
                    progress.pool(version_progress_name, message=f'Формирование файлов')  # Добавить конкретику
                    for i, result in enumerate(results):
                        progress.pool(version_progress_name, percent=ceil(i / len(data) * 100))
                        if decamelize(parameters[put_id][col_name]['type']) in PATH_TYPE_LIST:
                            for j in range(len(result['instructions'])):
                                result['instructions'][j] = result['instructions'][j].replace(
                                    str(version_sources_path), '')[1:]
                        instructions += result['instructions']
                        result_params = result['parameters']
                        # classes_names += result['parameters']['classes_names']
                        # if put_id == put_data[0].id and parameters['type'] != LayerOutputTypeChoice.Classification:
                        # self.y_cls += [os.path.basename(os.path.dirname(data_to_pass[i])) for _ in
                        # range(len(result['instructions']))]

                instructions_data = InstructionsData(instructions=instructions, parameters=result_params)
                instructions_data.parameters.update({'put_type': decamelize(parameters[put_id][col_name]['type'])})
                put_instructions[put_id].update({col_name: instructions_data})
                tags[put_id].update({col_name: decamelize(parameters[put_id][col_name]['type'])})

        return put_instructions, tags

    def create_instructions(self, version_data, sources_temp_directory, version_paths_data):

        tags = {}

        inp_data, inp_parameters = self.collect_data_to_pass(
            put_data=version_data.inputs,
            processing=version_data.processing,
            sources_temp_directory=sources_temp_directory
        )

        inputs, inp_tags = self.create_put_instructions(
            dictio=inp_data,
            parameters=inp_parameters,
            version_sources_path=version_paths_data.sources
        )

        out_data, out_parameters = self.collect_data_to_pass(
            put_data=version_data.outputs,
            processing=version_data.processing,
            sources_temp_directory=sources_temp_directory
        )

        outputs, out_tags = self.create_put_instructions(
            dictio=out_data,
            parameters=out_parameters,
            version_sources_path=version_paths_data.sources
        )

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)
        tags.update(inp_tags)
        tags.update(out_tags)

        return instructions, tags

    def create_preprocessing(self, asd):
        print('Тут создаем препроцессинг и обучаем')

    def create_input_parameters(self, input_instr, version_data, preprocessing, version_paths_data):

        inputs = {}
        columns = {}
        for key in input_instr.keys():
            put_array = []
            columns[key] = {}
            for col_name, data in input_instr[key].items():
                data_to_pass = data.instructions[0]
                if data.parameters["put_type"] in PATH_TYPE_LIST:
                    data_to_pass = str(version_paths_data.sources.joinpath(data_to_pass))
                options_to_pass = data.parameters.copy()
                if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                    prep = preprocessing.preprocessing.get(key).get(col_name)
                    options_to_pass.update([('preprocess', prep)])

                array = self.multithreading_array([data_to_pass], [options_to_pass])[0]
                put_array.append(array)
                if options_to_pass.get('classes_names'):
                    classes_names = options_to_pass.get('classes_names')
                else:
                    classes_names = sorted(
                        [os.path.basename(x) for x in version_data.inputs.get(key).parameters.keys()])

                # Прописываем параметры для колонки
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'shape': array.shape,
                                  'name': version_data.inputs.get(key).name,
                                  'task': camelize(data.parameters.get('put_type')),
                                  'classes_names': classes_names,
                                  'classes_colors': data.parameters.get('classes_colors'),
                                  'num_classes': len(classes_names) if classes_names else 0,
                                  'encoding': 'none' if not data.parameters.get('encoding') else data.parameters.get(
                                      'encoding')}
                current_column = DatasetInputsData(**col_parameters)
                columns[key].update([(col_name, current_column.native())])

            put_array = np.concatenate(put_array, axis=0)
            classes_colors_list, classes_names_list, encoding_list, task_list = [], [], [], []
            for value in columns[key].values():
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
                              'name': version_data.inputs.get(key).name,
                              'task': task_list[0] if len(set(task_list)) == 1 else 'Dataframe',
                              'classes_names': classes_names_list if classes_names_list else None,
                              'classes_colors': classes_colors_list if classes_colors_list else None,
                              'num_classes': len(classes_names_list) if classes_names_list else None,
                              'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}

            inputs[key] = DatasetInputsData(**put_parameters).native()

        return inputs, columns

    def create_output_parameters(self, output_instr, version_data, preprocessing, version_paths_data):

        outputs = {}
        columns = {}
        for key in output_instr.keys():
            put_array = []
            columns[key] = {}
            for col_name, data in output_instr[key].items():
                data_to_pass = data.instructions[0]
                if data.parameters["put_type"] in PATH_TYPE_LIST:
                    data_to_pass = str(version_paths_data.sources.joinpath(data_to_pass))
                options_to_pass = data.parameters.copy()
                if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                    prep = preprocessing.preprocessing.get(key).get(col_name)
                    options_to_pass.update([('preprocess', prep)])

                array = self.multithreading_array([data_to_pass], [options_to_pass])[0]
                if not array.shape:
                    array = np.expand_dims(array, 0)
                put_array.append(array)

                if options_to_pass.get('classes_names'):
                    classes_names = options_to_pass.get('classes_names')
                else:
                    classes_names = sorted(
                        [os.path.basename(x) for x in version_data.outputs.get(key).parameters.keys()])

                # Прописываем параметры для колонки
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'shape': array.shape,
                                  'name': version_data.outputs.get(key).name,
                                  'task': camelize(data.parameters.get('put_type')),
                                  'classes_names': classes_names,
                                  'classes_colors': data.parameters.get('classes_colors'),
                                  'num_classes': len(classes_names) if classes_names else 0,
                                  'encoding': 'none' if not data.parameters.get('encoding') else data.parameters.get(
                                      'encoding')}
                current_column = DatasetOutputsData(**col_parameters)
                columns[key].update([(col_name, current_column.native())])

            put_array = np.concatenate(put_array, axis=0)
            classes_colors_list, classes_names_list, encoding_list, task_list = [], [], [], []
            for value in columns[key].values():
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
                              'name': version_data.outputs.get(key).name,
                              'task': task_list[0] if len(task_list) == 1 else 'Dataframe',
                              'classes_names': classes_names_list if classes_names_list else None,
                              'classes_colors': classes_colors_list if classes_colors_list else None,
                              'num_classes': len(classes_names_list) if classes_names_list else None,
                              'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}

            outputs[key] = DatasetOutputsData(**put_parameters).native()

        return outputs, columns

    def create_service_parameters(self, output_instr, version_data, preprocessing, version_paths_data):

        service = {}

        return service

    def create_dataset_arrays(self, put_data, version_paths_data, dataframe, preprocessing):

        for split in ['train', 'val']:
            open_mode = 'w' if not version_paths_data.arrays.joinpath('dataset.h5') else 'a'
            hdf = h5py.File(version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                data_to_pass = []
                dict_to_pass = []
                for i in range(0, len(dataframe[split])):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                            parameters_to_pass.update([('preprocess',
                                                        preprocessing.preprocessing.get(key).get(col_name))])
                        if parameters_to_pass['put_type'] in PATH_TYPE_LIST:
                            tmp_data.append(str(version_paths_data.sources.joinpath(dataframe[split].loc[i, col_name])))
                        else:
                            tmp_data.append(dataframe[split].loc[i, col_name])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(
                    version_progress_name,
                    message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                    percent=0
                )

                hdf[split].create_group(f'id_{key}')

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(self.multithreading_array, data_to_pass, dict_to_pass)
                    for i, result in enumerate(results):
                        progress.pool(version_progress_name, percent=ceil(i / len(data_to_pass) * 100))
                        array = np.concatenate(result, axis=0)
                        hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array)
                        del result
            hdf.close()
