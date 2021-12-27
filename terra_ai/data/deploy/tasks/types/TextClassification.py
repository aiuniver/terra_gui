import json
import os
import random
from pathlib import Path, PurePath
from typing import List, Tuple

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    actual: str
    data: List[Tuple[str, int]]


class DataList(DataBaseList):
    source_path: Path = PurePath()

    class Meta:
        source = Item

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
        predict_path = Path(self.path_deploy, "preset", "out")
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)
        predict_file = Path(predict_path, "predict.txt")

        for _index in indexes:
            self.update(_index)

        label = []
        for item in self.preset:
            label.append(json.dumps(item.data, ensure_ascii=False))
        with open(predict_file, "w") as predict_file_ref:
            predict_file_ref.write("\n".join(label))

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination_source = Path(self.source_path, f"{index + 1}.txt")
        with open(destination_source, "w") as destination_source_ref:
            destination_source_ref.write(json.dumps(item.source, ensure_ascii=False))


class Data(DataBase):
    class Meta:
        source = DataList
