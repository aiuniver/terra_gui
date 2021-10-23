import json
import random
from pathlib import Path
from typing import List, Any

from terra_ai.data.mixins import BaseMixinData
from terra_ai.settings import DEPLOY_PRESET_COUNT
from ..extra import DataBaseList, DataBase


class Item(BaseMixinData):
    source: dict
    predict: Any


class DataList(DataBaseList):
    class Meta:
        source = Item

    def reload(self, indexes: List[int] = None):
        if indexes is None:
            indexes = list(range(DEPLOY_PRESET_COUNT))
        indexes = list(filter(self._positive_int_filter, indexes))
        indexes = list(map(int, indexes))
        if not len(self):
            return

        for _index in indexes:
            item = random.choice(self)
            self.preset[_index] = item

        _presets = []
        for preset in self.preset:
            _presets.append(preset.native())

        with open(Path(self.path, "presets.json"), "w") as preset_file_ref:
            preset_file_ref.write(json.dumps(_presets, ensure_ascii=False))


class Data(DataBase):
    class Meta:
        source = DataList
