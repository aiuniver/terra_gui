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

        return np.array([0])

    def preprocess(self, array: np.ndarray, preprocess, **options):

        return array
