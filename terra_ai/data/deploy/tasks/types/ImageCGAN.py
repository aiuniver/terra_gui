import json
import random
from typing import List
from pathlib import Path

from PIL import Image

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: str
    actual: str


class DataList(DataBaseList):
    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({"source": str(Path(self.path_deploy, data.get("source")))})
        return data

    def reload(self, indexes: List[int] = None):
        super(DataList, self).reload()
        if indexes is None:
            range_count = (
                DEPLOY_PRESET_COUNT if DEPLOY_PRESET_COUNT <= len(self) else len(self)
            )
            indexes = list(range(range_count))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        label_file = Path(self.path_deploy, "label.txt")

        for _index in indexes:
            self.update(_index)

        label = []
        for item in self.preset:
            label.append(json.dumps(item.actual, ensure_ascii=False))
        with open(label_file, "w") as label_file_ref:
            label_file_ref.write("\n".join(label))

    def update(self, index: int):
        item = random.choice(self)
        self.preset[index] = item

        destination = Path(self.path_deploy, f"{index + 1}.jpg")
        Image.open(Path(self.path_deploy, item.source)).save(destination)


class Data(DataBase):
    class Meta:
        source = DataList
