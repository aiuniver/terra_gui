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

    DatasetCreateMissing = {
        "ru": "Ошибка формирования данных для датасета. Класс `%s`, метод `%s`.",
        "eng": "Error in generating data for the dataset. `%s` class, `%s` method.",
    }

    DatasetPrepareMissing = {
        "ru": "Ошибка подготовки датасета. Класс `%s`, метод `%s`.",
        "eng": "Dataset preparation error. `%s` class, `%s` method.",
    }

    ModelCreateMissing = {
        "ru": "Ошибка загрузки модели. Класс `%s`, метод `%s`.",
        "eng": "Model loading error. `%s` class, `%s` method.",
    }

    NoPredict = {
        "ru": "Отсутствует результат предсказания. Класс `%s`, метод `%s`.",
        "eng": "There is no prediction result. `%s` class, `%s` method.",
    }

    PredictionMissing = {
        "ru": "Ошибка получения предсказания. Класс `%s`, метод `%s`.",
        "eng": "Error in obtaining a prediction. `%s` class, `%s` method.",
    }

    PostprocessMissing = {
        "ru": "Ошибка постобработки результатов предсказания. Модуль `%s`, метод `%s`.",
        "eng": "Error in postprocessing prediction results. `%s` module, `%s` method.",
    }

    PresetsMissing = {
        "ru": "Невозможно получить демонстрационные результаты. Класс `%s`, метод `%s`.",
        "eng": "It is impossible to get demo results. `%s` class, `%s` method.",
    }

    NoTrainedModelMissing = {
        "ru": "Выбранная модель не обучена. Класс `%s`, метод `%s`.",
        "eng": "The selected model is not trained. `%s` class, `%s` method.",
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


class DatasetCreateException(DeployException):
    class Meta:
        message: dict = DeployMessages.DatasetCreateMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class DatasetPrepareException(DeployException):
    class Meta:
        message: dict = DeployMessages.DatasetPrepareMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class ModelCreateException(DeployException):
    class Meta:
        message: dict = DeployMessages.ModelCreateMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class NoPredictException(DeployException):
    class Meta:
        message: dict = DeployMessages.NoPredict

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class PredictionException(DeployException):
    class Meta:
        message: dict = DeployMessages.PredictionMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class PostprocessException(DeployException):
    class Meta:
        message: dict = DeployMessages.PostprocessMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class PresetsException(DeployException):
    class Meta:
        message: dict = DeployMessages.PresetsMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


class NoTrainedModelException(DeployException):
    class Meta:
        message: dict = DeployMessages.NoTrainedModelMissing

    def __init__(self, __method: str, __class: str, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)