from typing import Any
from abc import ABC, abstractmethod


class BaseService(ABC):

    @abstractmethod
    def execute(self, sources: Any):
        pass


class Service(BaseService):

    def __init__(self):
        self.speech2text = SpeechToText

    def execute(self, sources: Any):
        pass

    def get(self, input_type):
        return self.__getattribute__(input_type)()


class SpeechToText(Service):

    def execute(self, sources):
        if isinstance(sources, str):
            sources = [sources]

        for source in sources:
            yield source
