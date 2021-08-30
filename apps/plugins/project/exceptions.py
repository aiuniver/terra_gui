from enum import Enum


class ExceptionMessages(str, Enum):
    UnknownError = "Unknown error"
    DatasetModelInputsCountNotMatch = "Количество входных слоев датасета не совпадает с количеством входных слоев редактируемой модели"
    DatasetModelOutputsCountNotMatch = "Количество выходных слоев датасета не совпадает с количеством выходных слоев редактируемой модели"


class ProjectException(Exception):
    class Meta:
        message = ExceptionMessages.UnknownError

    def __init__(self, *args, **kwargs):
        super().__init__(self.Meta.message.value, *args, **kwargs)


class DatasetModelInputsCountNotMatchException(ProjectException):
    class Meta:
        message = ExceptionMessages.DatasetModelInputsCountNotMatch


class DatasetModelOutputsCountNotMatchException(ProjectException):
    class Meta:
        message = ExceptionMessages.DatasetModelOutputsCountNotMatch
