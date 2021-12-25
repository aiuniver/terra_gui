from enum import Enum

from .base import TerraBaseException


class CallbacksMessages(dict, Enum):
    Undefined = {
        "ru": "Неопределенная ошибка обучения",
        "eng": "Undefined error of training",
    }
    ErrorInModuleInMethod = {
        "ru": "Ошибка в модуле `%s`, метод `%s`: %s",
        "eng": "Error in module `%s` method `%s`: %s",
    }
    ErrorInClassInMethod = {
        "ru": "Ошибка в классе `%s`, метод `%s`: %s",
        "eng": "Error in class `%s` method `%s`: %s",
    }
    SetInteractiveAttributesMissing = {
        "ru": "Не удалось установить аттрибуты для интерактивного коллбэка. Класс `%s`, метод `%s`.",
        "eng": "The attributes for the interactive callback could not be set. `%s` class, `%s` method.",
    }


class UndefinedException(TerraBaseException):
    class Meta:
        message: dict = CallbacksMessages.Undefined


class ErrorInModuleInMethodException(TerraBaseException):
    class Meta:
        message: dict = CallbacksMessages.ErrorInModuleInMethod

    def __init__(self, __module_name: str, __method_name: str, __error: str, **kwargs):
        super().__init__(__module_name, __method_name, __error, **kwargs)


class ErrorInClassInMethodException(TerraBaseException):
    class Meta:
        message: dict = CallbacksMessages.ErrorInClassInMethod

    def __init__(self, __class_name: str, __method_name: str, __error: str, **kwargs):
        super().__init__(__class_name, __method_name, __error, **kwargs)


class SetInteractiveAttributesException(TerraBaseException):
    class Meta:
        message: dict = CallbacksMessages.SetInteractiveAttributesMissing

    def __init__(self, __module_name: str, __method_name: str, **kwargs):
        super().__init__(__module_name, __method_name, **kwargs)