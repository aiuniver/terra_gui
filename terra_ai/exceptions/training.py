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
    TrainingAlreadyExists = {
        "ru": "Обучение `%s` уже существует",
        "eng": "Training `%s` already exists",
    }
    TrainingDefaultName = {
        "ru": "Запрещено использовать название `%s`",
        "eng": "It is forbidden to use the name `%s`",
    }

    NoCheckpointParameters = {
        "ru": "Отсутствуют параметры для чекпойнта. Класс `%s`, метод `%s`.",
        "eng": "There are no parameters for the checkpoint. `%s` class, `%s` method.",
    }

    NoCheckpointMetric = {
        "ru": "Для чекпойнта выбран тип 'Метрика', но метрика не установлена. Класс `%s`, метод `%s`.",
        "eng": "The 'Metric' type is selected for the checkpoint, but the metric is not set. `%s` class, `%s` method.",
    }

    NoImportantParameters = {
        "ru": "Отсутствуют важные параметры: `%s`. Класс `%s`, метод `%s`.",
        "eng": "Important parameters are missing: `%s`. `%s` class, `%s` method.",
    }

    PredictImpossible = {
        "ru": "Невозможно получить предсказание. Класс `%s`, метод `%s`.",
        "eng": "It is impossible to get a prediction. `%s` class, `%s` method.",
    }

    StartNumBatchesMissing = {
        "ru": "Ошибка расчета батчей на старте обучения. Класс `%s`, метод `%s`.",
        "eng": "Error in calculating the batches at the start of training. `%s` class, `%s` method.",
    }

    BatchResultMissing = {
        "ru": "Ошибка обучения на батче № %s. Класс `%s`, метод `%s`.",
        "eng": "Learning error on the batch № %s. `%s` class, `%s` method.",
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


class TrainingAlreadyExistsException(TrainingException):
    class Meta:
        message = TrainingMessages.TrainingAlreadyExists


class TrainingDefaultNameException(TrainingException):
    class Meta:
        message = TrainingMessages.TrainingDefaultName


class NoCheckpointParameters(TrainingException):
    class Meta:
        message = TrainingMessages.NoCheckpointParameters

    def __init__(self, __module: str, __method: str, **kwargs):
        super().__init__(str(__module), str(__method), **kwargs)


class NoCheckpointMetric(TrainingException):
    class Meta:
        message = TrainingMessages.NoCheckpointMetric

    def __init__(self, __module: str, __method: str, **kwargs):
        super().__init__(str(__module), str(__method), **kwargs)


class NoImportantParameters(TrainingException):
    class Meta:
        message = TrainingMessages.NoImportantParameters

    def __init__(self, __params: str, __module: str, __method: str, **kwargs):
        super().__init__(str(__params), str(__module), str(__method), **kwargs)


class PredictImpossible(TrainingException):
    class Meta:
        message = TrainingMessages.PredictImpossible

    def __init__(self, __module: str, __method: str, **kwargs):
        super().__init__(str(__module), str(__method), **kwargs)


class StartNumBatchesMissing(TrainingException):
    class Meta:
        message = TrainingMessages.StartNumBatchesMissing

    def __init__(self, __module: str, __method: str, **kwargs):
        super().__init__(str(__module), str(__method), **kwargs)


class BatchResultMissing(TrainingException):
    class Meta:
        message = TrainingMessages.BatchResultMissing

    def __init__(self, __butch_number: int, __module: str, __method: str, **kwargs):
        super().__init__(str(__butch_number), str(__module), str(__method), **kwargs)

