import json
import random
import shutil
from pathlib import Path, PosixPath
from typing import List, Tuple


from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: PosixPath
    actual: str
    data: List[Tuple[str, float]]


class DataList(DataBaseList):
    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({"source": str(Path(self.path_model, data.get("source")))})
        return data

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        label_file = Path(self.path_deploy, "label.txt")

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
            item.source, Path(self.path_deploy, f"{index + 1}{item.source.suffix}")
        )


class Data(DataBase):
    class Meta:
        source = DataList
