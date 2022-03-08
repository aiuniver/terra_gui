import numpy as np

# from tensorflow.keras import utils
from typing import Any

# from terra_ai.data.datasets.extra import LayerScalerImageChoice
from .base import Array


class TimeseriesArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):

        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):

        return np.array(source)

    def preprocess(self, array: np.ndarray, preprocess, **options):

        orig_shape = array.shape
        array = preprocess.transform(array.reshape(-1, 1))
        array = array.reshape(orig_shape)

        return array
