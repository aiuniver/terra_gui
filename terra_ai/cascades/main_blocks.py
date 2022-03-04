from abc import ABC, abstractmethod

import numpy as np


class BaseBlock(ABC):

    def __init__(self):
        self.inputs: dict = {}

    @abstractmethod
    def execute(self, *args):
        pass


class CascadeBlock:

    def get(self, type_, **kwargs):
        if kwargs:
            return self.__getattribute__(type_)(**kwargs)
        else:
            return self.__getattribute__(type_)


class ModelOut(BaseBlock, CascadeBlock):

    def execute(self, model_predict: np.ndarray, options: dict):
        pass
