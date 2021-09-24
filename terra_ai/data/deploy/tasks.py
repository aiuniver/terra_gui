import os
import random
import shutil

from typing import List, Optional
from pathlib import Path

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.deploy.extra import CollectionTypeChoice
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.training.guinn import interactive


class BaseCollectionList(List):
    _path: Optional[Path]

    def __init__(self, *args, path: Path = None, **kwargs):
        self._path = path
        if self._path is not None:
            os.makedirs(self._path, exist_ok=True)
        super().__init__(*args, **kwargs)

    def _clear(self):
        self.clear()
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            os.makedirs(self._path, exist_ok=True)

    def reload(self, range_indexes: List):
        raise MethodNotImplementedException("reload", self.__class__.__name__)


class ImageClassificationCollectionList(BaseCollectionList):
    def reload(self, range_indexes: List):
        source = interactive.deploy_presets_data
        # --- Temporary: delete before merge branch -------------------
        # with open("/home/bl146u/Downloads/ddd1.json", "r") as json_ref:
        #     source = json.load(json_ref)
        # -------------------------------------------------------------
        if not source:
            self._clear()
            return

        for index in range_indexes:
            try:
                value = self[index]
            except IndexError:
                continue
            if value is not None:
                # remove files by index
                print(self[index])
            value = source[random.randint(0, len(source))]
            self[index] = value


class BaseCollection(BaseMixinData):
    type: CollectionTypeChoice
    data: BaseCollectionList
