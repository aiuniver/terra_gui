import re
import random

from pathlib import Path
from typing import Any, List

from terra_ai.settings import DEPLOY_PRESET_COUNT
from terra_ai.exceptions.deploy import MethodNotImplementedException

from . import types


class DeployBase:
    _path: Path = None
    _available: List[Any] = []
    _data: List[Any] = []

    def __init__(self, path: Path, available: List[Any] = None):
        self._path = Path(path)
        self.available = available

    @property
    def type(self):
        return re.sub(r"^Deploy", "", self.__class__.__name__)

    @property
    def exists(self):
        return len(self._data) > 0

    @property
    def available(self) -> List:
        return self._available

    @available.setter
    def available(self, values):
        if not isinstance(values, List):
            values = []
        _dataclass = getattr(types, self.type).Data
        self._available = list(map(lambda item: _dataclass(**item), values))
        self._data = [None] * DEPLOY_PRESET_COUNT
        self.reload()

    def _positive_int_filter(self, value) -> int:
        try:
            value = int(value)
        except ValueError:
            return False
        return value in list(range(DEPLOY_PRESET_COUNT))

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not self._available:
            return
        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        raise MethodNotImplementedException("update", self.__class__.__name__)

    def dict(self) -> dict:
        return {
            "type": self.type,
            "exists": self.exists,
            "data": list(map(lambda item: item.native() if item else None, self._data)),
        }


class DeployImageSegmentation(DeployBase):
    pass


class DeployImageClassification(DeployBase):
    def update(self, index: int):
        value = random.choice(self.available)
        self._data[index] = value
        for item in self._data:
            if not item:
                continue
            print(item.data)


class DeployTextSegmentation(DeployBase):
    def update(self, index: int):
        value = random.choice(self.available)
        self._data[index] = value


class DeployTextClassification(DeployBase):
    pass
