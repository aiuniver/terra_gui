import os
import random
from pathlib import Path, PurePath
from typing import List, Tuple

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    format: str


class DataList(DataBaseList):
    source_path: Path = PurePath()
    format_path: Path = PurePath()

    class Meta:
        source = Item

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.source_path = Path(self.path_deploy, "preset", "in")
        self.format_path = Path(self.path_deploy, "preset", "out")
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.format_path, exist_ok=True)

        for _index in indexes:
            self.update(_index)

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination_source = Path(self.source_path, f"{index + 1}.txt")
        destination_format = Path(self.format_path, f"{index + 1}.txt")

        with open(destination_source, "w") as destination_source_ref:
            destination_source_ref.write(item.source)
        with open(destination_format, "w") as destination_format_ref:
            destination_format_ref.write(item.format)


class Data(DataBase):
    color_map: List[Tuple[str, str, Tuple[int, int, int]]]

    class Meta:
        source = DataList
