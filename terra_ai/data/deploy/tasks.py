from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.dataset import DatasetPathsData


class BaseCollection(BaseMixinData):
    def __init__(self, dataset: DatasetData, **data):
        super().__init__(**data)


class ImageClassificationCollection(BaseCollection):
    pass


class ImageSegmentationCollection(BaseCollection):
    pass
