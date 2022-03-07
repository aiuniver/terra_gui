from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Array(ABC):

    @abstractmethod
    def prepare(self, sources: list, **options):
        pass

    @abstractmethod
    def create(self, source: Any, **options):
        pass

    @abstractmethod
    def preprocess(self, array: np.ndarray, preprocess):
        pass
