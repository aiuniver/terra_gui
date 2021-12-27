import os
import random
import shutil

from pathlib import Path, PurePath
from typing import List

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    predict: str


class DataList(DataBaseList):
    source_path: Path = PurePath()
    predict_path: Path = PurePath()

    class Meta:
        source = Item

    def preset_update(self, data):
        source_path = str(Path(self.path_deploy, data.get("source")))
        with open(source_path) as source_ref:
            data.update(
                {
                    "source": source_ref.read(),
                    "predict": str(Path(self.path_deploy, data.get("predict"))),
                }
            )
        return data

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            range_count = (
                DEPLOY_PRESET_COUNT if DEPLOY_PRESET_COUNT <= len(self) else len(self)
            )
            indexes = list(range(range_count))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.source_path = Path(self.path_deploy, "preset", "in")
        self.predict_path = Path(self.path_deploy, "preset", "out")
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.predict_path, exist_ok=True)

        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination_source = Path(self.source_path, f"{index + 1}.txt")
        destination_predict = Path(self.predict_path, f"{index + 1}.webm")

        shutil.copyfile(Path(self.path_deploy, item.source), destination_source)
        shutil.copyfile(Path(self.path_deploy, item.predict), destination_predict)


class Data(DataBase):
    class Meta:
        source = DataList
