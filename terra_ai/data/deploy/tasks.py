from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.dataset import DatasetPathsData


class BaseCollection(BaseMixinData):
    path: DatasetPathsData

    def __init__(self, dataset: DatasetData, **data):
        super().__init__(**data)

    def dict(self, **kwargs):
        kwargs.update(
            {
                "exclude": {"path"},
            }
        )
        return super().dict(**kwargs)


class ImageClassificationCollection(BaseCollection):
    pass
