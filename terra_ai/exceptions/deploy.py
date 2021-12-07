from enum import Enum

from .base import TerraBaseException


class DeployMessages(dict, Enum):
    Undefined = {
        "ru": "Неопределенная ошибка деплоя",
        "eng": "Undefined error of deploy",
    }
    RequestAPI = {
        "ru": "Ошибка запроса API при деплое сервера",
        "eng": "Error API request to deploy server",
    }
    MethodNotImplemented = {
        "ru": "Метод `%s` должен быть реализован в классе `%s`",
        "eng": "Method `%s` must be implemented in class `%s`",
    }


class DeployException(TerraBaseException):
    class Meta:
        message: dict = DeployMessages.Undefined


class RequestAPIException(DeployException):
    class Meta:
        message: dict = DeployMessages.RequestAPI


class MethodNotImplementedException(DeployException):
    class Meta:
        message: dict = DeployMessages.MethodNotImplemented

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)
