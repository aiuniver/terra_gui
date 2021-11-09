import random

from typing import List, Tuple
from pathlib import PosixPath
from pydantic import PositiveFloat

from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: PosixPath
    actual: str
    data: List[Tuple[str, PositiveFloat]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def update(self, index: int):
        value = random.choice(self)
        self.preset[index] = value


class Data(DataBase):
    class Meta:
        source = DataList
