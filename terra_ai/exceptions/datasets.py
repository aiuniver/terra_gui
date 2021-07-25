from enum import Enum

from .base import TerraBaseException


class DatasetsMessages(str, Enum):
    pass


class DatasetsException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of datasets"
