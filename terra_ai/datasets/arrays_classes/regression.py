import numpy as np

from typing import Any

from .base import Array
from ...data.datasets.extra import LayerScalerImageChoice


class RegressionArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):

        return np.array([source])

    def preprocess(self, array: np.ndarray, preprocess, **options):

        array = preprocess.transform(array.reshape(-1, 1))[0]

        return array
