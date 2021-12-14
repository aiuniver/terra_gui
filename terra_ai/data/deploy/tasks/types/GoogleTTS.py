from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    pass


class DataList(DataBaseList):
    class Meta:
        source = Item


class Data(DataBase):
    class Meta:
        source = DataList
