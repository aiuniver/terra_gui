from enum import Enum

from .base import TerraBaseException


class ProfileMessages(dict, Enum):
    Undefined = {"ru": "Неопределенная ошибка профиля",
                 "eng": "Undefined error of deploy"}


class ProfileException(TerraBaseException):
    class Meta:
        message: dict = ProfileMessages.Undefined
