import numpy as np

from typing import Any

from .base import Array


class DiscriminatorArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        lst = [None for _ in range(len(sources))]

        instructions = {'instructions': lst, 'parameters': options}

        return instructions

    def create(self, source: Any = None, **options):

        if source is None:
            source = np.array([1])

        instructions = {'instructions': source, 'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        return array