import os
import random
import shutil

from typing import List, Optional
from pathlib import Path

from terra_ai import settings
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
        self._reset()

    def _reset(self):
        self.clear()
        for _ in list(range(settings.DEPLOY_PRESET_COUNT)):
            self.append(None)
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            os.makedirs(self._path, exist_ok=True)

    def try_init(self):
        if not list(filter(None, self)) and interactive.deploy_presets_data:
            self.reload(list(range(settings.DEPLOY_PRESET_COUNT)))

    def reload(self, range_indexes: List):
        raise MethodNotImplementedException("reload", self.__class__.__name__)


class ImageClassificationCollectionList(BaseCollectionList):
    def reload(self, range_indexes: List):
        source = interactive.deploy_presets_data
        if not source:
            self._reset()
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
            except Exception:
                pass
            value = source[random.randint(0, len(source) - 1)]
            filepath = Path(value.get("source"))
            destination = Path(self._path, f"{index+1}{filepath.suffix}")
            shutil.copyfile(filepath.absolute(), destination)
            value.update({"source": str(destination.absolute())})
            self[index] = value


class BaseCollection(BaseMixinData):
    type: CollectionTypeChoice
    data: BaseCollectionList

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        data.update({"data": self.data})
        return data
