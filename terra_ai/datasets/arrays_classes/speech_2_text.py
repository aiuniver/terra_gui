import numpy as np

from typing import Any

from .base import Array


class Speech2TextArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):

        return_list = ['no_data' for _ in range(len(sources))]
        instructions = {'instructions': return_list,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):
        instructions = {'instructions': np.array([0]),
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        return array
