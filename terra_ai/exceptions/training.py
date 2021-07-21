from enum import Enum

from .base import TerraBaseException


class TrainingMessages(str, Enum):
    pass


class TrainingException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of training"
