from enum import Enum

from .base import TerraBaseException


class TrainingMessages(dict, Enum):
    Undefined = {
        "ru": "Неопределенная ошибка обучения",
        "eng": "Undefined error of training",
    }
    TooBigBatchSize = {
        "ru": "batch_size `%s` не может быть больше чем размер тренировочной выборки `%s`",
        "eng": "batch_size `%s` can't be bigger than training sample size `%s`",
    }
    FileNotFound = {
        "ru": "Файл или директория отсутствует по пути: %s",
        "eng": "No such file or directory: %s",
    }
    MethodNotImplemented = {
        "ru": "Метод `%s` должен быть реализован в классе `%s`",
        "eng": "Method `%s` must be implemented in class `%s`",
    }


class TrainingException(TerraBaseException):
    class Meta:
        message: dict = TrainingMessages.Undefined


class TooBigBatchSize(TrainingException):
    class Meta:
        message = TrainingMessages.TooBigBatchSize

    def __init__(self, __bath_size: int, __train_size: int, **kwargs):
        super().__init__(str(__bath_size), str(__train_size), **kwargs)


class FileNotFoundException(TrainingException):
    class Meta:
        message = TrainingMessages.FileNotFound


class MethodNotImplementedException(TrainingException):
    class Meta:
        message: dict = TrainingMessages.MethodNotImplemented

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)
