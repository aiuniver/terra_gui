import numpy as np

from typing import Any

from .base import Array


class TrackerArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):
        instructions = {'instructions': np.array([source]),
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        return array
