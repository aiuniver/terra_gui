import numpy as np
import pandas as pd

from ast import literal_eval
from typing import Any

from .base import Array


class RawArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        pass

    def create(self, source: Any = None, **options):
        if isinstance(source, str):
            try:
                source = literal_eval(source)
            except:
                pass
            source = np.array([source])
        elif isinstance(source, list):
            source = np.array(source)
        elif isinstance(source, pd.Series):
            source = source.values
        else:
            source = np.array([source])

        instructions = {'instructions': source,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):
        return array
