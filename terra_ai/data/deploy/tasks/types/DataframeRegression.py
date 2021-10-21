import random

from typing import List, Tuple
from pydantic import PositiveFloat, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: List[PositiveInt, str, str]
    actual: str
    data: List[Tuple[str, PositiveFloat]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def update(self, index: int):
        value = random.choice(self)
        self.preset[index] = value


class Data(DataBase):
    columns: List[str]
    predict_column: str

    class Meta:
        source = DataList
