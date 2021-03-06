import os
import json
import shutil
import random

from typing import List, Tuple
from pathlib import Path

from terra_ai.data.types import ConstrainedFloatValueGe0Le100
from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    actual: str
    data: List[Tuple[str, ConstrainedFloatValueGe0Le100]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({"source": str(Path(self.path_deploy, data.get("source")))})
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
        self.actual_path = Path(self.path_deploy, "preset", "out")
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.actual_path, exist_ok=True)

        label_file = Path(self.actual_path, "label.txt")

        for _index in indexes:
            self.update(_index)

        label = []
        for item in self.preset:
            label.append(json.dumps(item.data, ensure_ascii=False))
        with open(label_file, "w") as label_file_ref:
            label_file_ref.write("\n".join(label))

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        shutil.copyfile(
            Path(self.path_deploy, item.source),
            Path(self.source_path, f"{index + 1}{Path(item.source).suffix}"),
        )


class Data(DataBase):
    class Meta:
        source = DataList
