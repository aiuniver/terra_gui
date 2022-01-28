import random
from pathlib import Path

from PIL import Image

from terra_ai.data.mixins import BaseMixinData
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str


class DataList(DataBaseList):
    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({"source": str(Path(self.path_deploy, data.get("source")))})
        return data

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination = Path(self.path_deploy, f"{index + 1}.jpg")
        Image.open(Path(self.path_deploy, item.source)).save(destination)


class Data(DataBase):
    class Meta:
        source = DataList
