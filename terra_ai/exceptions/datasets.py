from enum import Enum

from .base import TerraBaseException


class DatasetsMessages(dict, Enum):
    Undefined = {"ru": "Неопределенная ошибка датасетов",
                 "eng": "Undefined error of datasets"}
    DatasetKeras = {"ru": "Неизвестный Keras-у датасет `%s`",
                    "eng": "Unknown keras dataset `%s`"}
    DatasetSourceLoadUndefinedMethod = {"ru": "Метод загрузки источников датасета в режиме `%s` еще не определен",
                                        "eng": "Undefined method for loading dataset sources in mode `%s`"}
    DatasetChoiceUndefinedMethod = {"ru": "Метод выбора датасета из группы `%s` еще не определен",
                                    "eng": "Undefined method for choice dataset from group `%s`"}
    UnknownDataset = {"ru": "Группа `%s` не имеет датасет с именем `%s`",
                      "eng": "Unknown `%s` dataset `%s`"}


class DatasetsException(TerraBaseException):
    class Meta:
        message: dict = DatasetsMessages.Undefined


class DatasetSourceLoadUndefinedMethodException(DatasetsException):
    class Meta:
        message: dict = DatasetsMessages.DatasetSourceLoadUndefinedMethod

    def __init__(self, __mode: str, **kwargs):
        super().__init__(str(__mode), **kwargs)


class DatasetChoiceUndefinedMethodException(DatasetsException):
    class Meta:
        message: dict = DatasetsMessages.DatasetChoiceUndefinedMethod

    def __init__(self, __group: str, **kwargs):
        super().__init__(str(__group), **kwargs)


class UnknownDatasetException(DatasetsException):
    class Meta:
        message: dict = DatasetsMessages.UnknownDataset

    def __init__(self, __group: str, __name: str, **kwargs):
        super().__init__(str(__group), str(__name), **kwargs)
