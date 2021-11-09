import json
import os
import random
from pathlib import Path, PurePath, PosixPath
from typing import List, Tuple

from PIL import Image

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: PosixPath
    segment: PosixPath
    data: List[Tuple[str, Tuple[int, int, int]]]


class DataList(DataBaseList):
    source_path: Path = PurePath()
    segment_path: Path = PurePath()

    class Meta:
        source = Item

    def preset_update(self, data):
        data.update({k: str(Path(self.path_model, data.get(k)))} for k in ('source', 'segment'))
        return data

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        self.source_path = Path(self.path, "preset", "in")
        self.segment_path = Path(self.path, "preset", "out")
        os.makedirs(self.source_path, exist_ok=True)
        os.makedirs(self.segment_path, exist_ok=True)
        label_file = Path(self.path, "label.txt")

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

        destination_source = Path(self.source_path, f"{index + 1}.jpg")
        destination_segment = Path(self.segment_path, f"{index + 1}.jpg")

        Image.open(item.source).save(destination_source)
        Image.open(item.segment).save(destination_segment)


class Data(DataBase):
    class Meta:
        source = DataList
