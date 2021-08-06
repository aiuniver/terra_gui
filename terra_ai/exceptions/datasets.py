from enum import Enum

from .base import TerraBaseException


class DatasetsMessages(str, Enum):
    DatasetKeras = "Unknown keras dataset `%s`"
    DatasetSourceLoadUndefinedMethod = (
        "Undefined method for loading dataset sources in mode `%s`"
    )
    DatasetChoiceUndefinedMethod = "Undefined method for choice dataset from group `%s`"
    UnknownKerasDataset = "Unknown keras dataset `%s`"
    UnknownCustomDataset = "Unknown custom dataset `%s`"


class DatasetsException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of datasets"


class DatasetSourceLoadUndefinedMethodException(DatasetsException):
    class Meta:
        message: str = DatasetsMessages.DatasetSourceLoadUndefinedMethod

    def __init__(self, __mode: str, *args):
        super().__init__(self.Meta.message % str(__mode), *args)


class DatasetChoiceUndefinedMethodException(DatasetsException):
    class Meta:
        message: str = DatasetsMessages.DatasetChoiceUndefinedMethod

    def __init__(self, __group: str, *args):
        super().__init__(self.Meta.message % str(__group), *args)


class UnknownKerasDatasetException(DatasetsException):
    class Meta:
        message: str = DatasetsMessages.UnknownKerasDataset

    def __init__(self, __name: str, *args):
        super().__init__(self.Meta.message % str(__name), *args)


class UnknownCustomDatasetException(DatasetsException):
    class Meta:
        message: str = DatasetsMessages.UnknownCustomDataset

    def __init__(self, __name: str, *args):
        super().__init__(self.Meta.message % str(__name), *args)
