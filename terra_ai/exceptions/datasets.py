from enum import Enum

from .base import TerraBaseException


class DatasetsMessages(str, Enum):
    DatasetKeras = "Unknown keras dataset `%s`"
    DatasetSourceLoadUndefinedMethod = (
        "Undefined method for loading dataset sources in mode `%s`"
    )


class DatasetsException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of datasets"


class DatasetSourceLoadUndefinedMethodException(DatasetsException):
    class Meta:
        message: str = DatasetsMessages.DatasetSourceLoadUndefinedMethod

    def __init__(self, __mode: str, *args):
        super().__init__(self.Meta.message % str(__mode), *args)
