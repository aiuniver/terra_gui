import numpy as np

from typing import Any

from .base import Array


class TrackerArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):

        return_list = ['no_data' for _ in range(len(sources))]
        instructions = {'instructions': return_list,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):

        array = np.array([0], dtype=np.uint8)

        return array

    def preprocess(self, array: np.ndarray, preprocess, **options):

        return array
