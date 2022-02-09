import numpy as np

from typing import Any

from .base import Array


class ScalerArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):
        instructions = {'instructions': np.array(source),
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        if array.shape != ():
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        else:
            array = options['preprocess'].transform(array.reshape(-1, 1))[0]

        return array
