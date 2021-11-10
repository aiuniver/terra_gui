from enum import Enum


class ExceptionMessages(str, Enum):
    UnknownError = "Unknown error"
    DatasetModelInputsCountNotMatch = "Количество входных слоев датасета не совпадает с количеством входных слоев редактируемой модели"
    DatasetModelOutputsCountNotMatch = "Количество выходных слоев датасета не совпадает с количеством выходных слоев редактируемой модели"
    ProjectAlreadyExists = "Проект `%s` уже существует"


class ProjectException(Exception):
    class Meta:
        message = ExceptionMessages.UnknownError

    def __init__(self, *args, **kwargs):
        msg = self.Meta.message.value

        if args:
            msg = msg % args

        super().__init__(msg)


class DatasetModelInputsCountNotMatchException(ProjectException):
    class Meta:
        message = ExceptionMessages.DatasetModelInputsCountNotMatch


class DatasetModelOutputsCountNotMatchException(ProjectException):
    class Meta:
        message = ExceptionMessages.DatasetModelOutputsCountNotMatch


class ProjectAlreadyExistsException(ProjectException):
    class Meta:
        message = ExceptionMessages.ProjectAlreadyExists
