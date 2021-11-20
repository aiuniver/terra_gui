import os
import random
from pathlib import Path, PurePath, PosixPath
from typing import List

from PIL import Image

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    predict: str


class DataList(DataBaseList):
    preset_path: Path = PurePath()
    predict_path: Path = PurePath()

    class Meta:
        source = Item

    def preset_update(self, data):
        data.update(
            {k: str(Path(self.path_deploy, data.get(k)))} for k in ("source", "predict")
        )
        return data

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.preset_path = Path(self.path_deploy, "preset", "in")
        self.predict_path = Path(self.path_deploy, "preset", "out")
        os.makedirs(self.preset_path, exist_ok=True)
        os.makedirs(self.predict_path, exist_ok=True)

        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination_source = Path(self.preset_path, f"{index + 1}.jpg")
        destination_predict = Path(self.predict_path, f"{index + 1}.jpg")

        Image.open(Path(self.path_deploy, item.source)).save(destination_source)
        Image.open(Path(self.path_deploy, item.predict)).save(destination_predict)


class Data(DataBase):
    class Meta:
        source = DataList
