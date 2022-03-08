import numpy as np

from typing import Any
from tensorflow import random as tf_random

from .base import Array


class NoiseArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        lst = [None for _ in range(len(sources))]
        instructions = {'instructions': lst, 'parameters': options}

        return instructions

    def create(self, source: Any = None, **options):

        array = tf_random.normal(shape=options['shape']).numpy()

        return array

    def preprocess(self, array: np.ndarray, preprocess, **options):

        return array
