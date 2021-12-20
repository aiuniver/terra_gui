import json
import random
from pathlib import Path, PurePath
from typing import List, Tuple, Union

from pydantic import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: List[Union[PositiveInt, str, str]]
    actual: str
    data: List[Tuple[str, float]]


class DataList(DataBaseList):
    preset_file: Path = PurePath()

    class Meta:
        source = Item

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.preset_file = Path(self.path_deploy, "preset.txt")
        label_file = Path(self.path_deploy, "label.txt")

        for _path in (self.preset_file, label_file):
            if _path.exists():
                _path.unlink()

        for _index in indexes:
            self.update(_index)

        label = []
        for item in self.preset:
            label.append(json.dumps(item.data, ensure_ascii=False))
        with open(label_file, "a") as label_file_ref:
            label_file_ref.write("\n".join(label))

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        with open(self.preset_file, "a") as preset_file_ref:
            preset_file_ref.write(json.dumps(item.source, ensure_ascii=False))
            preset_file_ref.write("\n")


class Data(DataBase):
    columns: List[str]
    predict_column: str

    class Meta:
        source = DataList
