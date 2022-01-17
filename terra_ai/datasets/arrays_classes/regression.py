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
        instructions = {'instructions': np.array([source]),
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            array = options['preprocess'].transform(array.reshape(-1, 1))[0]

        return array
