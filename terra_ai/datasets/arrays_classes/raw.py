import numpy as np
import pandas as pd

from ast import literal_eval
from typing import Any

from .base import Array


class RawArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any = None, **options):
        if isinstance(source, str):
            try:
                source = literal_eval(source)
            except:
                pass
            array = np.array([source])
        elif isinstance(source, list):
            array = np.array(source)
        elif isinstance(source, pd.Series):
            array = source.values
        else:
            array = np.array([source])

        return array

    def preprocess(self, array: np.ndarray, preprocess, **options):

        return array
