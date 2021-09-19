from pathlib import Path
from typing import Optional

from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy.extra import TaskTypeChoice
from terra_ai.data.deploy.collection import CollectionData


class Collection:
    __path: Optional[Path] = None
    __dataset: Optional[DatasetData] = None
    __type: Optional[TaskTypeChoice] = None

    @property
    def data(self) -> Optional[CollectionData]:
        if not self.__type:
            return
        return CollectionData(type=self.__type)

    def __define_type(self):
        self.__type = TaskTypeChoice.image_segmentation

    def __clear(self):
        self.__dataset = None
        self.__path = None
        self.__type = None

    def update(
        self, dataset: Optional[DatasetData] = None, path: Optional[Path] = None
    ):
        self.__clear()
        if not dataset or not path:
            return

        self.__dataset = dataset
        self.__path = path

        self.__define_type()
