from enum import Enum

from terra_ai.exceptions.base import TerraBaseException


class DatasetsMessages(dict, Enum):
    Undefined = {
        "ru": "Неопределенная ошибка датасетов",
        "eng": "Undefined error of datasets",
    }
    SourceLoadUndefinedMethod = {
        "ru": "Метод загрузки источников датасета в режиме `%s` еще не определен",
        "eng": "Undefined method for loading dataset sources in mode `%s`",
    }
    ChoiceUndefinedMethod = {
        "ru": "Метод выбора датасета из группы `%s` еще не определен",
        "eng": "Undefined method for choice dataset from group `%s`",
    }
    UndefinedGroup = {
        "ru": "Не определена группа датасетов `%s`",
        "eng": "Datasets group `%s` is undefined",
    }
    NotFoundInGroup = {
        "ru": "Не найден датасет `%s` в группе `%s`",
        "eng": "Dataset `%s` not found in group `%s`",
    }
    UndefinedConfig = {
        "ru": "Не определен файл конфигурации датасета `%s` группы `%s`",
        "eng": "Dataset's `%s` file configure of group `%s` is undefined",
    }
    CanNotBeDeleted = {
        "ru": "Датасет `%s` из группы `%s` нельзя удалить",
        "eng": "The dataset `%s` cannot be deleted from the group `%s`",
    }


class DatasetException(TerraBaseException):
    class Meta:
        message: dict = DatasetsMessages.Undefined


class DatasetSourceLoadUndefinedMethodException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.SourceLoadUndefinedMethod

    def __init__(self, __mode: str, **kwargs):
        super().__init__(str(__mode), **kwargs)


class DatasetChoiceUndefinedMethodException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.ChoiceUndefinedMethod

    def __init__(self, __group: str, **kwargs):
        super().__init__(str(__group), **kwargs)


class DatasetUndefinedGroupException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.UndefinedGroup

    def __init__(self, __group: str, **kwargs):
        super().__init__(str(__group), **kwargs)


class DatasetNotFoundInGroupException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.NotFoundInGroup

    def __init__(self, __name: str, __group: str, **kwargs):
        super().__init__(str(__name), str(__group), **kwargs)


class DatasetUndefinedConfigException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.UndefinedConfig

    def __init__(self, __name: str, __group: str, **kwargs):
        super().__init__(str(__name), str(__group), **kwargs)


class DatasetCanNotBeDeletedException(DatasetException):
    class Meta:
        message: dict = DatasetsMessages.CanNotBeDeleted

    def __init__(self, __dataset: str, __group: str, **kwargs):
        super().__init__(str(__dataset), str(__group), **kwargs)
