from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetOutputsData, DatasetInputsData
from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass, multithreading_array
from terra_ai.datasets.data import DataType
from terra_ai.settings import VERSION_PROGRESS_NAME
from math import ceil
import numpy as np
import h5py
import concurrent.futures

from terra_ai.utils import camelize


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


class TimeseriesClass(PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        length, depth, step = 0, 0, 0
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'Timeseries':
                if out_data.parameters.options.trend:
                    out_data.parameters.options.depth = 1
                length = out_data.parameters.options.length
                depth = out_data.parameters.options.depth
                step = out_data.parameters.options.step
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type in ['Classification', 'Scaler', 'Raw']:
                inp_data.parameters.options.length = length
                inp_data.parameters.options.depth = depth
                inp_data.parameters.options.step = step

        return version_data

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
                                  'shape': array.shape,
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
                                  'shape': array.shape,
                                  'name': col_name,
                                  'task': camelize(parameters_to_pass.get('put_type')),
                                  'classes_names': None,
                                  'classes_colors': None,
                                  'num_classes': parameters_to_pass.get('num_classes'),
                                  'encoding': parameters_to_pass.get('encoding', 'none')}
                current_column = DatasetOutputsData(**col_parameters)
                columns[key].update({col_name: current_column.native()})
            put_array = np.concatenate(put_array, axis=1)
            out_parameters = {'datatype': DataType.get(len(put_array.shape), 'DIM'),
                              'dtype': str(put_array.dtype),
                              'shape': put_array.shape,
                              'name': f'Выход {key}',
                              'task': 'Timeseries',
                              'classes_names': None,
                              'classes_colors': None,
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
                depth, length, step = 0, 0, 1
                for col_name, data in put_data[key].items():
                    depth = data.parameters['depth'] if 'depth' in data.parameters.keys() and \
                                                        data.parameters['depth'] else 0
                    length = data.parameters['length'] if depth else 0
                    step = data.parameters['step'] if depth else 1
                for i in range(0, len(dataframe[split]) - length - depth, step):
                    tmp_data = []
                    tmp_parameter_data = []
                    for col_name, data in put_data[key].items():
                        parameters_to_pass = data.parameters.copy()
                        if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                            parameters_to_pass.update([('preprocess',
                                                        preprocessing.preprocessing.get(key).get(col_name))])
                        tmp_data.append(dataframe[split].loc[i:i + length - 1, col_name])
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
