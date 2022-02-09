import concurrent.futures
import os
import h5py
from math import ceil

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice
from terra_ai.settings import VERSION_PROGRESS_NAME
from terra_ai.datasets.data import DataType
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names
from terra_ai.utils import decamelize, camelize
from terra_ai.datasets.creating_classes.base import BaseClass, multithreading_array, PreprocessingNumericClass


class YoloV4Class(PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):
        version_data = kwargs['version_data']
        version_path_data = kwargs['version_path_data']
        version_data.processing['1'].parameters.frame_mode = version_data.processing['0'].parameters.image_mode
        names_list = get_od_names(version_data, version_path_data)
        version_data.processing['1'].parameters.classes_names = names_list
        version_data.processing['1'].parameters.num_classes = len(names_list)

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
                        if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                            prep = preprocessing.preprocessing.get(key).get(col_name)
                            parameters_to_pass.update([('preprocess', prep)])
                        if parameters_to_pass['put_type'] in PATH_TYPE_LIST:
                            tmp_data.append(os.path.join(version_paths_data.sources, dataframe[split].loc[i, col_name]))
                        elif parameters_to_pass['put_type'] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            tmp_data.append(dataframe[split].loc[i, col_name])
                            height, width = dataframe[split].iloc[i, 0].split(';')[1].split(',')
                            parameters_to_pass.update([('orig_x', int(width)), ('orig_y', int(height))])
                        tmp_parameter_data.append(parameters_to_pass)
                    data_to_pass.append(tmp_data)
                    dict_to_pass.append(tmp_parameter_data)

                progress.pool(VERSION_PROGRESS_NAME,
                              message=f'Формирование массивов {split.title()} выборки. ID: {key}.', percent=0)
                if parameters_to_pass['put_type'] == decamelize(LayerOutputTypeChoice.ObjectDetection):
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
                        if not parameters_to_pass['put_type'] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                            # array = np.concatenate(result, axis=0)
                            hdf[f'{split}/id_{key}'].create_dataset(str(i), data=result[0])
                        else:
                            for n in range(3):
                                hdf[f'{split}/id_{key + n}'].create_dataset(str(i), data=result[0][n])
                                hdf[f'{split}/id_{key + n}_service'].create_dataset(str(i), data=result[0][n + 3])
                        del result
            hdf.close()
