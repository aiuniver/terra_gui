import random

from typing import List, Tuple
from pydantic import FilePath, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: FilePath
    segment: FilePath
    data: List[Tuple[str, Tuple[PositiveInt]]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def update(self, index: int):
        value = random.choice(self)
        self.preset[index] = value


class Data(DataBase):
    class Meta:
        source = DataList
