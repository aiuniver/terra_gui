from pathlib import Path

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.dataset import DatasetPathsData


class BaseCollection(BaseMixinData):
    def __init__(self, dataset: DatasetData, path: Path, **data):
        # print(dataset)
        # print(path)
        # print(DatasetPathsData(basepath=path))
        super().__init__(**data)


class ImageClassificationCollection(BaseCollection):
    pass
