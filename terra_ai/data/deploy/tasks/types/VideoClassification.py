import random

from typing import List, Tuple
from pathlib import Path
from pydantic import PositiveFloat

from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    actual: str
    data: List[Tuple[str, PositiveFloat]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({"source": str(Path(self.path_deploy, data.get("source")))})
        return data

    def update(self, index: int):
        value = random.choice(self)
        self.preset[index] = value


class Data(DataBase):
    class Meta:
        source = DataList
