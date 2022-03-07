import concurrent.futures
import os

import h5py
import numpy as np
import pandas as pd
from itertools import repeat
from math import ceil
from pathlib import Path

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetOutputsData, DatasetInputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerPrepareMethodChoice, LayerScalerImageChoice, \
    LayerSelectTypeChoice, LayerTypeChoice
from terra_ai.datasets import arrays_classes
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.settings import VERSION_PROGRESS_NAME
from terra_ai.datasets.data import InstructionsData, DataType, DatasetInstructionsData
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names, get_image_size
from terra_ai.logging import logger
from terra_ai.utils import decamelize, camelize, autodetect_encoding


def multithreading_instructions(one_path, params, dataset_folder, col_name, idx):

    try:
        instruction = getattr(getattr(arrays_classes, decamelize(params["type"])),
                              f'{params["type"]}Array')().prepare(
            sources=[one_path],
            dataset_folder=dataset_folder,
            **params['options'],
            **{'cols_names': col_name, 'put': idx}
        )

    except Exception:
        progress.pool(VERSION_PROGRESS_NAME, error=f'Ошибка создания инструкций для {col_name}')
        logger.debug(f'Создание инструкций провалилось на {one_path}')
        raise

    return instruction


def multithreading_array(row, instructions, preprocess):

    full_array = []
    for h in range(len(row)):
        try:
            array = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
                            f'{camelize(instructions[h]["put_type"])}Array')().create(source=row[h], **instructions[h])
            if preprocess:
                array = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
                                f'{camelize(instructions[h]["put_type"])}Array')().preprocess(array, preprocess)
            full_array.append(array)
        except Exception:
            progress.pool(VERSION_PROGRESS_NAME, error='Ошибка создания массивов данных')
            raise

    return full_array


# def multithreading_preprocessing(array, instructions):
#
#     full_array = []
#     for h in range(len(row)):
#         try:
#             # create = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
#             #                  f'{camelize(instructions[h]["put_type"])}Array')().create(
#             #     source=row[h], **instructions[h])
#             prepr = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
#                             f'{camelize(instructions[h]["put_type"])}Array')().preprocess(
#                 create['instructions'], **create['parameters'])
#             full_array.append(prepr)
#         except Exception:
#             progress.pool(VERSION_PROGRESS_NAME, error='Ошибка создания массивов данных')
#             raise
#
#     return full_array


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


class BaseClass(object):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        return version_data

    @staticmethod
    def collect_data_to_pass(put_data, sources_temp_directory, put_idx):

        def get_data(handler, preprocess, tree, idx):

            data_to_pass = {}
            parameters_to_pass = {}
            preprocess_to_pass = {}

            for up_id in handler.bind.up:
                data_obj = tree.get(up_id)
                data_to_pass[idx] = {}
                parameters_to_pass[idx] = {}
                preprocess_to_pass[idx] = {}
                if data_obj.parameters.type == LayerSelectTypeChoice.folder:
                    collected_data = []
                    for folder_name in data_obj.parameters.data:
                        current_path = Path(sources_temp_directory).joinpath(folder_name)  # sources_temp_directory
                        for direct, folder, files_name in os.walk(current_path):
                            if files_name:
                                for file_name in sorted(files_name):
                                    collected_data.append(os.path.join(current_path, file_name))
                    print(collected_data[:5])
                    print(handler.parameters.native())
                    data_to_pass[idx].update(
                        {f'{idx}_{data_obj.name}': collected_data}
                    )
                    parameters_to_pass[idx].update(
                        {f'{idx}_{data_obj.name}': handler.parameters.native()})
                    preprocess_to_pass[put_idx].update({f'{put_idx}_{data_obj.name}': preprocess})
                elif data_obj.parameters.type == LayerSelectTypeChoice.table:
                    current_path = Path(sources_temp_directory).joinpath(data_obj.parameters.file)
                    _, enc = autodetect_encoding(str(current_path), True)
                    for column in data_obj.parameters.data:
                        collected_data = pd.read_csv(current_path, sep=None, usecols=[column],
                                                     engine='python', encoding=enc).loc[:, column] \
                            .to_list()
                        if decamelize(decamelize(handler.parameters.type)) in PATH_TYPE_LIST:
                            collected_data = [str(Path(sources_temp_directory).joinpath(Path(x))) for x
                                              in collected_data]
                        print(collected_data[:5])
                        print(handler.parameters.native())
                        data_to_pass[idx].update({f'{idx}_{column}': collected_data})
                        parameters_to_pass[idx].update(
                            {f'{idx}_{column}': handler.parameters.native()}
                        )
                        preprocess_to_pass[put_idx].update({f'{put_idx}_{column}': preprocess})

            return data_to_pass, parameters_to_pass, preprocess_to_pass

        full_data = {}
        full_parameters = {}
        full_preprocess = {}
        for layer in put_data:
            if layer.type in [LayerTypeChoice.input, LayerTypeChoice.output]:
                put_idx += 1
                full_data[put_idx] = {}
                full_parameters[put_idx] = {}
                full_preprocess[put_idx] = {}
                for element_id in layer.bind.up:
                    element = put_data.get(element_id)
                    if element.type == LayerTypeChoice.preprocess:
                        for handler_id in element.bind.up:
                            handler = put_data.get(handler_id)
                            data, parameters, preprocess = get_data(handler, element.parameters.native(),
                                                                    put_data, put_idx)
                            full_data[put_idx].update(data[put_idx])
                            full_parameters[put_idx].update(parameters[put_idx])
                            full_preprocess[put_idx].update(preprocess[put_idx])
                    elif element.type == LayerTypeChoice.handler:
                        data, parameters, preprocess = get_data(element, {},
                                                                put_data, put_idx)
                        full_data[put_idx].update(data[put_idx])
                        full_parameters[put_idx].update(parameters[put_idx])
                        full_preprocess[put_idx].update(preprocess[put_idx])

        return full_data, full_parameters, full_preprocess

    @staticmethod
    def create_put_instructions(dictio, parameters, preprocess, version_sources_path):

        put_instructions = {}
        tags = {}

        for put_id, pass_data_parameters in dictio.items():
            put_instructions[put_id] = {}
            tags[put_id] = {}
            for col_name, data in pass_data_parameters.items():
                instructions = []
                classes_names = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(multithreading_instructions,
                                           data,
                                           repeat(parameters[put_id][col_name]),
                                           repeat(version_sources_path),
                                           repeat(col_name),
                                           repeat(put_id))
                    progress.pool(VERSION_PROGRESS_NAME, message=f'Формирование файлов')  # Добавить конкретику
                    for i, result in enumerate(results):
                        progress.pool(VERSION_PROGRESS_NAME, percent=ceil(i / len(data) * 100))
                        if decamelize(parameters[put_id][col_name]['type']) in PATH_TYPE_LIST:
                            for j in range(len(result['instructions'])):
                                result['instructions'][j] = result['instructions'][j].replace(
                                    str(version_sources_path), '')[1:]
                        instructions += result['instructions']
                        result_params = result['parameters']
                        if parameters[put_id][col_name]['type'] == LayerOutputTypeChoice.Classification:
                            classes_names += result['parameters'].get('classes_names')

                column_name = ','.join(col_name.split(':')) if ':' in col_name else col_name
                # '\/:*?"<>|'
                instructions_data = InstructionsData(
                    instructions=instructions,
                    parameters=result_params,
                    preprocess=preprocess
                )
                if parameters[put_id][col_name]['type'] == LayerOutputTypeChoice.Classification:
                    instructions_data.parameters.update({'classes_names': list(set(classes_names))})
                    instructions_data.parameters.update({'num_classes': len(list(set(classes_names)))})
                instructions_data.parameters.update({'put_type': decamelize(parameters[put_id][col_name]['type'])})
                instructions_data.parameters.update({'col_name': column_name})
                print(instructions_data.parameters)

                put_instructions[put_id].update({column_name: instructions_data})
                tags[put_id].update({column_name: decamelize(parameters[put_id][col_name]['type'])})

        return put_instructions, tags

    def create_instructions(self, version_data, sources_temp_directory, version_paths_data):

        inp_data, inp_parameters, inp_preprocess = self.collect_data_to_pass(
            put_data=version_data.inputs,
            sources_temp_directory=sources_temp_directory,
            put_idx=0
        )

        inputs, inp_tags = self.create_put_instructions(
            dictio=inp_data,
            parameters=inp_parameters,
            preprocess=inp_preprocess,
            version_sources_path=version_paths_data.sources
        )

        out_data, out_parameters, out_preprocess = self.collect_data_to_pass(
            put_data=version_data.outputs,
            sources_temp_directory=sources_temp_directory,
            put_idx=len(inp_data)
        )

        outputs, out_tags = self.create_put_instructions(
            dictio=out_data,
            parameters=out_parameters,
            preprocess=out_preprocess,
            version_sources_path=version_paths_data.sources
        )

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)

        tags = {}
        tags.update(inp_tags)
        tags.update(out_tags)

        return instructions, tags

    @staticmethod
    def create_numeric_preprocessing(instructions):

        return {}

    @staticmethod
    def create_text_preprocessing(instructions):

        return {}

    @staticmethod
    def fit_numeric_preprocessing(put_data, preprocessing, sources_temp_directory):

        return preprocessing

    @staticmethod
    def fit_text_preprocessing(put_data, preprocessing, sources_temp_directory):

        return preprocessing

    @staticmethod
    def create_input_parameters(input_instr, version_data, preprocessing, version_paths_data):

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
                preprocess_to_pass = preprocessing[key].get(col_name)
                # if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                #     prep = preprocessing.preprocessing.get(key).get(col_name)
                #     options_to_pass.update([('preprocess', prep)])

                array = multithreading_array([data_to_pass], [options_to_pass], preprocess_to_pass)[0]
                put_array.append(array)
                if options_to_pass.get('classes_names'):
                    classes_names = options_to_pass.get('classes_names')
                else:
                    column = ' '.join(col_name.split('_')[1:])
                    for block in version_data.inputs:
                        if block.name == column or \
                                block.type == LayerTypeChoice.data and column in block.parameters.data:
                            classes_names = block.parameters.data

                # Прописываем параметры для колонки
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'col_type': type(data_to_pass).__name__,
                                  'shape': array.shape,
                                  'name': col_name,  # version_data.inputs.get(key).name,
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
                              'name': f'Вход {key}',
                              'task': task_list[0] if len(set(task_list)) == 1 else 'Dataframe',
                              'classes_names': classes_names_list if classes_names_list else None,
                              'classes_colors': classes_colors_list if classes_colors_list else None,
                              'num_classes': len(classes_names_list) if classes_names_list else None,
                              'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}

            inputs[key] = DatasetInputsData(**put_parameters).native()

        return inputs, columns

    @staticmethod
    def create_output_parameters(output_instr, version_data, preprocessing, version_paths_data):

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
                preprocess_to_pass = preprocessing[key].get(col_name)
                # if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                #     prep = preprocessing.preprocessing.get(key).get(col_name)
                #     options_to_pass.update([('preprocess', prep)])

                array = multithreading_array([data_to_pass], [options_to_pass], preprocess_to_pass)[0]

                if not array.shape:
                    array = np.expand_dims(array, 0)
                put_array.append(array)

                if options_to_pass.get('classes_names'):
                    classes_names = options_to_pass.get('classes_names')
                else:
                    column = ' '.join(col_name.split('_')[1:])
                    for block in version_data.outputs:
                        if block.name == column or \
                                block.type == LayerTypeChoice.data and column in block.parameters.data:
                            classes_names = block.parameters.data

                # Прописываем параметры для колонки
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'col_type': type(data_to_pass).__name__,
                                  'shape': array.shape,
                                  'name': col_name,
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
                              'name':  f'Выход {key}',
                              'task': task_list[0] if len(task_list) == 1 else 'Dataframe',
                              'classes_names': classes_names_list if classes_names_list else None,
                              'classes_colors': classes_colors_list if classes_colors_list else None,
                              'num_classes': len(classes_names_list) if classes_names_list else None,
                              'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}

            outputs[key] = DatasetOutputsData(**put_parameters).native()

        return outputs, columns

    @staticmethod
    def create_service_parameters(output_instr, version_data, preprocessing, version_paths_data):

        service = {}

        return service

    def create_arrays(self, instructions, version_paths_data, dataframe, preprocessing):

        self.create_put_arrays(
            put_data=instructions.inputs,
            version_paths_data=version_paths_data,
            dataframe=dataframe,
            preprocessing=preprocessing
        )
        self.create_put_arrays(
            put_data=instructions.outputs,
            version_paths_data=version_paths_data,
            dataframe=dataframe,
            preprocessing=preprocessing
        )

    @staticmethod
    def create_put_arrays(put_data, version_paths_data, dataframe, preprocessing):

        for split in ['train', 'val']:
            open_mode = 'w' if not version_paths_data.arrays.joinpath('dataset.h5') else 'a'
            hdf = h5py.File(version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                data_to_pass = []
                dict_to_pass = []
                parameters_to_pass = {}
                preprocess_to_pass = None
                for i in range(0, len(dataframe[split])):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        preprocess_to_pass = preprocessing.get(key).get(col_name)
                        if parameters_to_pass['put_type'] == 'noise':
                            continue
                        # if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                        #     parameters_to_pass.update([('preprocess',
                        #                                 preprocessing.preprocessing.get(key).get(col_name))])
                        if parameters_to_pass['put_type'] in PATH_TYPE_LIST:
                            tmp_data.append(str(version_paths_data.sources.joinpath(dataframe[split].loc[i, col_name])))
                        else:
                            tmp_data.append(dataframe[split].loc[i, col_name])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)
                if parameters_to_pass['put_type'] == 'noise':
                    continue

                progress.pool(
                    VERSION_PROGRESS_NAME,
                    message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                    percent=0
                )

                hdf[split].create_group(f'id_{key}')

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(multithreading_array, data_to_pass, dict_to_pass, repeat(preprocess_to_pass))
                    for i, result in enumerate(results):
                        progress.pool(VERSION_PROGRESS_NAME, percent=ceil(i / len(data_to_pass) * 100))
                        array = np.concatenate(result, axis=0)
                        hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array)
                        del result
            hdf.close()


class ClassificationClass(object):

    def __init__(self):

        self.y_cls = []

    def create_put_instructions(self, dictio, parameters, preprocess, version_sources_path):

        put_instructions = {}
        tags = {}

        for put_id, pass_data_parameters in dictio.items():
            put_instructions[put_id] = {}
            tags[put_id] = {}
            for col_name, data in pass_data_parameters.items():
                instructions = []
                classes_names = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(multithreading_instructions,
                                           data,
                                           repeat(parameters[put_id][col_name]),
                                           repeat(version_sources_path),
                                           repeat(col_name),
                                           repeat(put_id))
                    progress.pool(VERSION_PROGRESS_NAME, message=f'Формирование файлов')  # Добавить конкретику
                    for i, result in enumerate(results):
                        progress.pool(VERSION_PROGRESS_NAME, percent=ceil(i / len(data) * 100))
                        if decamelize(parameters[put_id][col_name]['type']) in PATH_TYPE_LIST:
                            for j in range(len(result['instructions'])):
                                result['instructions'][j] = result['instructions'][j].replace(
                                    str(version_sources_path), '')[1:]
                        instructions += result['instructions']
                        result_params = result['parameters']
                        if parameters[put_id][col_name]['type'] == LayerOutputTypeChoice.Classification:
                            classes_names += result['parameters'].get('classes_names')
                        else:
                            if put_id == 1:
                                self.y_cls += [Path(data[i]).parent.name for _ in range(len(result['instructions']))]
                instructions_data = InstructionsData(
                    instructions=instructions,
                    parameters=result_params,
                    preprocess=preprocess[put_id][col_name]
                )
                instructions_data.parameters.update({'put_type': decamelize(parameters[put_id][col_name]['type'])})
                column_name = ','.join(col_name.split(':')) if ':' in col_name else col_name
                if parameters[put_id][col_name]['type'] == LayerOutputTypeChoice.Classification:
                    instructions_data.instructions = self.y_cls
                    if Path(classes_names[0]).is_file():
                        for i in range(len(classes_names)):
                            classes_names[i] = Path(classes_names[i]).parent.name
                    instructions_data.parameters.update({'classes_names': list(set(classes_names))})
                    instructions_data.parameters.update({'num_classes': len(list(set(classes_names)))})
                elif parameters[put_id][col_name]['type'] in [LayerOutputTypeChoice.Tracker,
                                                              LayerOutputTypeChoice.Speech2Text,
                                                              LayerOutputTypeChoice.Text2Speech]:
                    instructions_data.instructions = ['no_data' for _ in range(len(self.y_cls))]
                instructions_data.parameters.update({'cols_names': column_name})
                print(instructions_data.parameters)
                put_instructions[put_id].update({column_name: instructions_data})
                tags[put_id].update({column_name: decamelize(parameters[put_id][col_name]['type'])})

        return put_instructions, tags


class YoloClass(object):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']
        source_path = kwargs['source_path']
        version_path_data = kwargs['version_path_data']

        image_mode = ''
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Image':
                image_mode = inp_data.parameters.options.image_mode
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type in ['YoloV3', 'YoloV4']:
                out_data.parameters.options.frame_mode = image_mode
                classes_names = get_od_names(version_data, source_path, version_path_data)
                out_data.parameters.options.num_classes = len(classes_names)
                out_data.parameters.options.classes_names = classes_names

        return version_data

    @staticmethod
    def create_output_parameters(output_instr, version_data, preprocessing, version_paths_data):

        outputs = {}
        columns = {}
        for key in output_instr.keys():
            columns[key] = {}
            for col_name, data in output_instr[key].items():
                data_to_pass = data.instructions[0]
                options_to_pass = data.parameters.copy()
                options_to_pass.update([('orig_x', 100), ('orig_y', 100)])
                array = multithreading_array([data_to_pass], [options_to_pass])[0]
                classes_names = options_to_pass.get('classes_names')
                for i in range(3):
                    col_parameters = {'datatype': DataType.get(len(array[i].shape), 'DIM'),
                                      'dtype': str(array[i].dtype),
                                      'col_type': type(data_to_pass).__name__,
                                      'shape': array[i].shape,
                                      'name': version_data.outputs.get(key).name,
                                      'task': camelize(options_to_pass.get('put_type')),
                                      'classes_names': classes_names,
                                      'classes_colors': None,
                                      'num_classes': len(classes_names) if classes_names else 0,
                                      'encoding': options_to_pass.get('encoding', 'none')}
                    current_column = DatasetOutputsData(**col_parameters)
                    columns[key + i] = {col_name: current_column.native()}
                    outputs[key + i] = current_column.native()

        return outputs, columns

    @staticmethod
    def create_service_parameters(output_instr, version_data, preprocessing, version_paths_data):

        service = {}
        for key in output_instr.keys():
            for col_name, data in output_instr[key].items():
                data_to_pass = data.instructions[0]
                options_to_pass = data.parameters.copy()
                options_to_pass.update([('orig_x', 100), ('orig_y', 100)])
                array = multithreading_array([data_to_pass], [options_to_pass])[0]
                classes_names = options_to_pass.get('classes_names')
                for i in range(3, 6):
                    put_parameters = {
                        'datatype': DataType.get(len(array[i].shape), 'DIM'),
                        'dtype': str(array[i].dtype),
                        'shape': array[i].shape,
                        'name': version_data.outputs.get(key).name,
                        'task': camelize(options_to_pass.get('put_type')),
                        'classes_names': classes_names,
                        'classes_colors': None,
                        'num_classes': len(classes_names) if classes_names else 0,
                        'encoding': options_to_pass.get('encoding', 'none')
                    }
                    service[key + i - 3] = DatasetOutputsData(**put_parameters)

        return service

    @staticmethod
    def create_put_arrays(put_data, version_paths_data, dataframe, preprocessing):

        for split in ['train', 'val']:
            open_mode = 'w' if not version_paths_data.arrays.joinpath('dataset.h5') else 'a'
            hdf = h5py.File(version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                data_to_pass = []
                dict_to_pass = []
                parameters_to_pass = {}
                for i in range(0, len(dataframe[split])):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        parameters_to_pass.update(data.preprocess['type'])
                        if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                            prep = preprocessing.preprocessing.get(key).get(col_name)
                            parameters_to_pass.update([('preprocess', prep)])
                        if parameters_to_pass['put_type'] in PATH_TYPE_LIST:
                            tmp_data.append(os.path.join(version_paths_data.sources, dataframe[split].loc[i, col_name]))
                        elif parameters_to_pass['put_type'] in [decamelize(LayerOutputTypeChoice.YoloV3),
                                                                decamelize(LayerOutputTypeChoice.YoloV4)]:
                            tmp_data.append(dataframe[split].loc[i, col_name])
                            height, width = get_image_size(
                                version_paths_data.sources.joinpath(dataframe[split].iloc[i, 0])
                            )
                            # height, width = dataframe[split].iloc[i, 0].split(';')[1].split(',')
                            parameters_to_pass.update([('orig_x', int(width)), ('orig_y', int(height))])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(VERSION_PROGRESS_NAME,
                              message=f'Формирование массивов {split.title()} выборки. ID: {key}.', percent=0)
                if parameters_to_pass['put_type'] in [decamelize(LayerOutputTypeChoice.YoloV3),
                                                      decamelize(LayerOutputTypeChoice.YoloV4)]:
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
                    results = executor.map(multithreading_array, data_to_pass, dict_to_pass)
                    for i, result in enumerate(results):
                        progress.pool(VERSION_PROGRESS_NAME, percent=ceil(i / len(data_to_pass) * 100))
                        if parameters_to_pass['put_type'] not in [decamelize(LayerOutputTypeChoice.YoloV3),
                                                                  decamelize(LayerOutputTypeChoice.YoloV4)]:
                            hdf[f'{split}/id_{key}'].create_dataset(str(i), data=result[0])
                        else:
                            for n in range(3):
                                hdf[f'{split}/id_{key + n}'].create_dataset(str(i), data=result[0][n])
                                hdf[f'{split}/id_{key + n}_service'].create_dataset(str(i), data=result[0][n + 3])
                        del result
            hdf.close()


class MainTimeseriesClass(object):

    @staticmethod
    def create_input_parameters(input_instr, version_data, preprocessing, version_paths_data):

        inputs = {}
        columns = {}
        for key in input_instr.keys():
            columns[key] = {}
            put_array = []
            parameters_to_pass = {}
            for col_name, data in input_instr[key].items():
                data_to_pass = data.instructions[:data.parameters['length']]
                parameters_to_pass = data.parameters.copy()
                if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                    parameters_to_pass.update([('preprocess',
                                                preprocessing.preprocessing.get(key).get(col_name))])
                array = multithreading_array([data_to_pass], [parameters_to_pass])
                array = postprocess_timeseries(array)
                put_array.append(array)
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'col_type': type(data_to_pass[0]).__name__,
                                  'shape': array.flatten().shape,
                                  'name': col_name,
                                  'task': camelize(parameters_to_pass.get('put_type')),
                                  'classes_names': parameters_to_pass.get('classes_names'),
                                  'classes_colors': None,
                                  'num_classes': parameters_to_pass.get('num_classes'),
                                  'encoding': parameters_to_pass.get('encoding', 'none')}
                current_column = DatasetInputsData(**col_parameters)
                columns[key].update({col_name: current_column.native()})
            put_array = np.concatenate(put_array, axis=1)
            out_parameters = {'datatype': DataType.get(len(put_array.shape), 'DIM'),
                              'dtype': str(put_array.dtype),
                              'shape': put_array.shape,
                              'name': f'Вход {key}',
                              'task': camelize(parameters_to_pass.get('put_type')),
                              'classes_names': parameters_to_pass.get('classes_names'),
                              'classes_colors': None,
                              'num_classes': parameters_to_pass.get('num_classes'),
                              'encoding': parameters_to_pass.get('encoding', 'none')}
            current_out = DatasetInputsData(**out_parameters)
            inputs[key] = current_out.native()

        return inputs, columns

    @staticmethod
    def create_output_parameters(output_instr, version_data, preprocessing, version_paths_data):

        outputs = {}
        columns = {}
        for key in output_instr.keys():
            columns[key] = {}
            put_array = []
            parameters_to_pass = {}
            for col_name, data in output_instr[key].items():
                data_to_pass = data.instructions[:data.parameters['length']]
                parameters_to_pass = data.parameters.copy()
                if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                    parameters_to_pass.update([('preprocess',
                                                preprocessing.preprocessing.get(key).get(col_name))])
                array = multithreading_array([data_to_pass], [parameters_to_pass])
                array = postprocess_timeseries(array)
                put_array.append(array)
                col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
                                  'dtype': str(array.dtype),
                                  'col_type': type(data_to_pass[0]).__name__,
                                  'shape': array.flatten().shape,
                                  'name': col_name,
                                  'task': camelize(parameters_to_pass.get('put_type')),
                                  'classes_names': parameters_to_pass.get('classes_names'),
                                  'classes_colors': parameters_to_pass.get('classes_colors'),
                                  'num_classes': parameters_to_pass.get('num_classes'),
                                  'encoding': parameters_to_pass.get('encoding', 'none')}
                current_column = DatasetOutputsData(**col_parameters)
                columns[key].update({col_name: current_column.native()})
            put_array = np.concatenate(put_array, axis=1)
            out_parameters = {'datatype': DataType.get(len(put_array.shape), 'DIM'),
                              'dtype': str(put_array.dtype),
                              'shape': put_array.shape,
                              'name': f'Выход {key}',
                              'task': camelize(parameters_to_pass.get('put_type')),
                              'classes_names': parameters_to_pass.get('classes_names'),
                              'classes_colors': parameters_to_pass.get('classes_colors'),
                              'num_classes': parameters_to_pass.get('num_classes'),
                              'encoding': parameters_to_pass.get('encoding', 'none')}
            current_out = DatasetOutputsData(**out_parameters)
            outputs[key] = current_out.native()

        return outputs, columns

    @staticmethod
    def create_put_arrays(put_data, version_paths_data, dataframe, preprocessing):

        for split in ['train', 'val']:
            open_mode = 'w' if not version_paths_data.arrays.joinpath('dataset.h5') else 'a'
            hdf = h5py.File(version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
            if split not in list(hdf.keys()):
                hdf.create_group(split)
            for key in put_data.keys():
                data_to_pass = []
                dict_to_pass = []
                depth, length, step = 0, 0, 0
                for col_name, data in put_data[key].items():
                    depth = data.parameters['depth']
                    length = data.parameters['length']
                    step = data.parameters['step']
                for i in range(0, len(dataframe[split]) - length - depth, step):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                            parameters_to_pass.update([('preprocess',
                                                        preprocessing.preprocessing.get(key).get(col_name))])
                        if data.parameters['put_type'] == 'timeseries_trend':
                            tmp_data.append([dataframe[split].loc[i, col_name],
                                             dataframe[split].loc[i + length, col_name]])
                        else:
                            tmp_data.append(dataframe[split].loc[
                                            i:i + length + depth - 1, col_name
                                            ])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(
                    VERSION_PROGRESS_NAME,
                    message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
                    percent=0
                )

                hdf[split].create_group(f'id_{key}')

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(multithreading_array, data_to_pass, dict_to_pass)
                    for i, result in enumerate(results):
                        progress.pool(VERSION_PROGRESS_NAME, percent=ceil(i / len(data_to_pass) * 100))
                        array = postprocess_timeseries(result)
                        hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array)
                        del result
            hdf.close()


class PreprocessingNumericClass(object):

    # @staticmethod
    # def create_numeric_preprocessing(instructions, preprocessing):
    #
    #     for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
    #         for col_name, data in put.items():
    #             preprocessing.create_scaler(**data.parameters)
    #
    #     return preprocessing

    @staticmethod
    def create_numeric_preprocessing(instructions):

        preprocess = {}
        merged_puts = instructions.inputs.copy()
        merged_puts.update(instructions.outputs.items())
        for put_id, data in merged_puts.items():
            preprocess[put_id] = {}
            for col_name, col_data in data.items():
                if col_data.preprocess:
                    preprocess[put_id].update(
                        {col_name:
                            getattr(CreatePreprocessing, f"create_{decamelize(col_data.preprocess['type'])}")(
                                col_data.preprocess['options'])
                         }
                    )

        return preprocess

    @staticmethod
    def fit_numeric_preprocessing(put_data, preprocessing, sources_temp_directory):

        for key in put_data.keys():
            for col_name, data in put_data[key].items():
                if data.preprocess:  # preprocessing.preprocessing[key][col_name]: #data.preprocess in data.parameters and data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler, None]:
                    progress.pool(VERSION_PROGRESS_NAME, message=f'Обучение {data.preprocess["type"]} для {col_name}')
                    data_length = len(data.instructions)
                    progress_count = 0
                    for i in range(data_length):
                        if data.parameters['put_type'] in PATH_TYPE_LIST:
                            data_to_pass = str(sources_temp_directory.joinpath(data.instructions[i]))
                        else:
                            data_to_pass = data.instructions[i]

                        array = getattr(getattr(arrays_classes, data.parameters["put_type"]),
                                        f'{camelize(data.parameters["put_type"])}Array')().create(
                            source=data_to_pass, **data.parameters)

                        if decamelize(data.preprocess['type']) == LayerScalerImageChoice.terra_image_scaler:
                            preprocessing.preprocessing[key][col_name].fit(array)
                        else:
                            preprocessing.preprocessing[key][col_name].fit(array.reshape(-1, 1))
                        progress_count += 1
                        progress.pool(VERSION_PROGRESS_NAME,
                                      message=f'Обучение {data.preprocess["type"]} для {col_name}',
                                      percent=ceil((progress_count / data_length) * 100)
                                      )

        return preprocessing


class PreprocessingTextClass(object):

    # @staticmethod
    # def create_text_preprocessing(instructions, preprocessing):
    #
    #     for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
    #         for col_name, data in put.items():
    #             if data.parameters['put_type'] == 'text':
    #                 if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
    #                                                          LayerPrepareMethodChoice.bag_of_words]:
    #                     preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
    #                 elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
    #                     preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)
    #
    #     return preprocessing

    @staticmethod
    def create_text_preprocessing(instructions):

        preprocess = {}
        merged_puts = instructions.inputs.copy()
        merged_puts.update(instructions.outputs.items())

        for put_id, data in merged_puts.items():
            preprocess[put_id] = {}
            for col_name, col_data in data.items():
                if col_data.preprocess:
                    preprocess[put_id].update(
                        {col_name: getattr(CreatePreprocessing,
                                           f"create_{decamelize(col_data.preprocess['type'])}")(
                            col_data.instructions, col_data.preprocess['options'])
                         }
                    )

        return preprocess

    # @staticmethod
    # def fit_text_preprocessing(put_data, preprocessing):

        # Из-за невозможности создания Word2Vec без сразу передачи в него корпусов текста, обучение текстовых
        # препроцессингов происходит сразу на этапе создания

        # for key in put_data.keys():
        #     for col_name, data in put_data[key].items():
        #         progress.pool(VERSION_PROGRESS_NAME, message=f'Обучение {camelize(data.parameters["scaler"])}')
        #         preprocessing.preprocessing[key][col_name].fit_on_texts(data.instructions)
        # pass
