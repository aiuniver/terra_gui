import json
import random
from pathlib import Path, PurePath
from typing import List

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    preset: list
    label: list


class DataList(DataBaseList):
    preset_file: Path = PurePath()
    label_file: Path = PurePath()

    class Meta:
        source = Item

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.preset_file = Path(self.path, "preset.txt")
        self.label_file = Path(self.path, "label.txt")

        for _path in (self.preset_file, self.label_file):
            _path.unlink(missing_ok=True)

        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        with open(self.preset_file, "a") as preset_file_ref:
            preset_file_ref.write(json.dumps(item.preset, ensure_ascii=False))
            preset_file_ref.write("\n")
        with open(self.label_file, "a") as label_file_ref:
            label_file_ref.write(json.dumps(item.label, ensure_ascii=False))
            label_file_ref.write("\n")


class Data(DataBase):
    columns: List[str]
    predict_column: str

    class Meta:
        source = DataList
