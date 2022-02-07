from terra_ai import progress
from terra_ai.data.datasets.extra import LayerPrepareMethodChoice, LayerScalerImageChoice
from terra_ai.datasets.creating import version_progress_name
from terra_ai.datasets.creating_classes.base import BaseClass
from terra_ai.datasets.utils import PATH_TYPE_LIST
from terra_ai.utils import camelize
import numpy as np


class ImageSegmentationClass(BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']
        version_data.processing['1'].parameters.height = version_data.processing['0'].parameters.height
        version_data.processing['1'].parameters.width = version_data.processing['0'].parameters.width

        return version_data

    @staticmethod
    def create_preprocessing(instructions, preprocessing):

        for put in instructions.inputs.values():
            for col_name, data in put.items():
                preprocessing.create_scaler(**data.parameters)

        return preprocessing

    def fit_preprocessing(self, put_data, preprocessing, sources_temp_directory):

        for key in put_data.keys():
            for col_name, data in put_data[key].items():
                if 'scaler' in data.parameters and \
                        data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler, None]:
                    progress.pool(version_progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
                    if data.parameters['put_type'] in PATH_TYPE_LIST:
                        for i in range(len(data.instructions)):

                            array = self.multithreading_array(
                                [str(sources_temp_directory.joinpath(data.instructions[i]))],
                                [data.parameters]
                            )[0]
                            if data.parameters['scaler'] == LayerScalerImageChoice.terra_image_scaler:
                                preprocessing.preprocessing[key][col_name].fit(array)
                            else:
                                preprocessing.preprocessing[key][col_name].fit(array.reshape(-1, 1))
                    else:
                        preprocessing.preprocessing[key][col_name].fit(
                            np.array(data.instructions).reshape(-1, 1))

        return preprocessing
