from enum import Enum

from .base import TerraBaseException


class TrainingMessages(dict, Enum):
    Undefined = {"ru": "Неопределенная ошибка обучения",
                 "eng": "Undefined error of training"}


class TrainingException(TerraBaseException):
    class Meta:
        message: dict = TrainingMessages.Undefined
