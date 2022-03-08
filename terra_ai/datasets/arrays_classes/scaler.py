import numpy as np

from typing import Any

from .base import Array


class ScalerArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):

        array = np.array(source)

        return array

    def preprocess(self, array: np.ndarray, preprocess, **options):

        if array.shape != ():
            orig_shape = array.shape
            array = preprocess.transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        else:
            array = preprocess.transform(array.reshape(-1, 1))[0]

        return array
