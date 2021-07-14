from enum import Enum

from .base import TerraBaseException


class ModelingMessages(str, Enum):
    pass


class ModelingException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of modeling"
