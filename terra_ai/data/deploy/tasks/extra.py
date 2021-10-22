from pathlib import Path
from typing import List, Any
from pydantic import validator, DirectoryPath

from terra_ai.data.mixins import BaseMixinData
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.settings import DEPLOY_PRESET_COUNT


class DataBaseList(List):
    path: DirectoryPath
    preset: List[Any] = []

    class Meta:
        source = None

    def __init__(self, *args):
        if len(args) > 1:
            self.path = args[1]
        self.preset = [None] * DEPLOY_PRESET_COUNT
        super().__init__(
            list(map(lambda item: self.Meta.source(**item), args[0])) if args else []
        )

    @property
    def presets(self) -> list:
        return list(map(lambda item: item.native() if item else None, self.preset))

    @staticmethod
    def _positive_int_filter(value) -> int:
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
        if not len(self):
            return
        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        raise MethodNotImplementedException("update", self.__class__.__name__)

    def dict(self):
        return list(map(lambda item: item.native() if item else None, self))


class DataBase(BaseMixinData):
    path: Path
    data: Any

    class Meta:
        source = DataBaseList

    @validator("data")
    def _validate_data(cls, value: DataBaseList, values) -> DataBaseList:
        data = cls.Meta.source(value, values.get("path"))
        data.reload()
        return data

    @property
    def presets(self) -> dict:
        data = self.native()
        data.update({"data": self.data.presets})
        return data

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"path"}})
        data = super().dict(**kwargs)
        data.update({"data": self.data.dict()})
        return data

    def reload(self, indexes: List[int] = None):
        self.data.reload(indexes)
