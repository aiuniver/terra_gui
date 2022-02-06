import concurrent.futures
import os
from itertools import repeat
from math import ceil
from pathlib import Path

from terra_ai import progress
from terra_ai.data.datasets.dataset import VersionPathsData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerODDatasetTypeChoice, LayerOutputTypeChoice, LayerEncodingChoice
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.creating import version_progress_name
from terra_ai.datasets.data import InstructionsData, DataType
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names
from terra_ai.logging import logger
from terra_ai.utils import decamelize
from terra_ai.datasets.creating_classes.base import BaseClass


class ImageObjectDetectionClass(BaseClass):

    def preprocess_version_data(self, **kwargs):
        version_data = kwargs['version_data']
        version_path_data = kwargs['version_path_data']
        version_data.processing['2'].parameters.frame_mode = version_data.processing['1'].parameters.image_mode
        names_list = get_od_names(version_data, version_path_data)
        version_data.processing['2'].parameters.classes_names = names_list
        version_data.processing['2'].parameters.num_classes = len(names_list)

        return version_data

    def create_service_parameters(self, output_instr, version_data, preprocessing, version_paths_data):

        # Пока сделано только для OD
        service = {}
        for key in output_instr.keys():
            for col_name, data in output_instr[key].items():
                if data.parameters['put_type'] == decamelize(LayerOutputTypeChoice.ObjectDetection):
                    data_to_pass = data.instructions[0]
                    if data.parameters["put_type"] in PATH_TYPE_LIST:
                        data_to_pass = str(version_paths_data.sources.joinpath(data_to_pass))

                    options_to_pass = data.parameters.copy()
                    if preprocessing.preprocessing.get(key) and preprocessing.preprocessing.get(key).get(col_name):
                        prep = preprocessing.preprocessing.get(key).get(col_name)
                        options_to_pass.update([('preprocess', prep)])

                    array = self.multithreading_array([data_to_pass], [options_to_pass])[0]
                    classes_names = options_to_pass.get('classes_names')
                    num_classes = len(classes_names) if classes_names else None
                    for n in range(3):
                        put_parameters = {
                            'datatype': DataType.get(len(array[n + 3].shape), 'DIM'),
                            'dtype': str(array[n + 3].dtype),
                            'shape': array[n + 3].shape,
                            'name': version_data.outputs.get(key).name,
                            'task': LayerOutputTypeChoice.ObjectDetection,
                            'classes_names': classes_names,
                            'classes_colors': None,
                            'num_classes': num_classes,
                            'encoding': LayerEncodingChoice.ohe
                        }
                        service[key + n] = DatasetOutputsData(**put_parameters)

        return service
