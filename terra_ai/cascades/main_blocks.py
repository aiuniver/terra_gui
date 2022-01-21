from abc import ABC, abstractmethod


class BaseBlock(ABC):

    def __init__(self):
        self.inputs: dict = {}

    @abstractmethod
    def execute(self):
        pass


class CascadeBlock:

    def get(self, type_, **kwargs):
        return self.__getattribute__(type_)(**kwargs)


class ModelBlock(BaseBlock, CascadeBlock):

    def execute(self):
        pass
