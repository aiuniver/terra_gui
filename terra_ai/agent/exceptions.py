from pathlib import Path
from typing import Any
from enum import Enum


class ExceptionMessages(dict, Enum):
    # Agent
    UnknownError = {"ru": "Неизвестная ошибка", "eng": "Unknown error"}
    CallMethodNotFound = {"ru": "Метод у `%s` должен иметь вид `%s`",
                          "eng": "Instance of `%s` must have method `%s`"}
    MethodNotCallable = {"ru": "Метод `%s` у `%s` должен быть вызываемым",
                         "eng": "Method `%s` of instance of `%s` must be callable"}
    FileNotFound = {"ru": "Файл или директория отсутствует по пути: %s",
                    "eng": "No such file or directory: %s"}
    # Project
    ProjectAlreadyExists = {"ru": "Проект `%s` уже существует",
                            "eng": "Project `%s` already exists"}
    ProjectNotFound = {"ru": "Проект `%s` не найден по пути: `%s`",
                       "eng": "Project `%s` not found in `%s`"}
    FailedGetProjectsInfo = {"ru": "Не удалось получить информацию о проекте. %s",
                             "eng": "Error when getting projects info: %s"}
    FailedSaveProject = {"ru": "Не удалось сохранить проект. %s",
                         "eng": "Error when saving project: %s"}
    FailedLoadProject = {"ru": "Не удалось загрузить проект. %s",
                         "eng": "Error when loading project: %s"}
    # Dataset
    FailedChoiceDataset = {"ru": "Не удалось выбрать датасет. %s",
                           "eng": "Error when choosing dataset: %s"}
    FailedDeleteDataset = {"ru": "Не удалось удалить датасет. %s",
                           "eng": "Dataset could not be deleted: %s"}
    DatasetCanNotBeDeleted = {"ru": "Датасет `%s` из группы `%s` не был удален",
                              "eng": "Dataset `%s` from group `%s` can't be deleted"}
    FailedGetProgressDatasetChoice = {"ru": "Не удалось получить прогресс выбора датасета. %s",
                                      "eng": "Could not get the progress of the dataset choice: %s"}
    FailedGetDatasetsInfo = {"ru": "Не удалось получить информацию о датасете. %s",
                             "eng": "Error when getting datasets info: %s"}
    FailedLoadDatasetsSource = {"ru": "Не удалось загрузить исходники датасета. %s",
                                "eng": "Error when loading datasets source: %s"}
    FailedLoadProgressDatasetsSource = {"ru": "Не удалось получить прогресс выбора датасета. %s",
                                        "eng": "Error when loading progress of datasets info: %s"}
    FailedGetDatasetsSources = {"ru": "Не удалось получить список исходников датасетов. %s",
                                "eng": "Could not get the datasets sources info: %s"}
    FailedCreateDataset = {"ru": "Не удалось создать датасет. %s",
                           "eng": "Could not create dataset. %s"}
    # Modeling
    ModelAlreadyExists = {"ru": "Модель `%s` уже существует",
                          "eng": "Model `%s` already exists"}
    FailedGetModelsList = {"ru": "Не удалось получить список моделей. %s",
                           "eng": "Could not get the models list: %s"}
    FailedGetModel = {"ru": "Не удалось получить модель. %s",
                      "eng": "Error when getting the model: %s"}
    FailedValidateModel = {"ru": "Не удалось провести валидацию модели. %s",
                           "eng": "Error when validating the model: %s"}
    FailedUpdateModel = {"ru": "Не удалось обновить модель. %s",
                         "eng": "Error when updating the model: %s"}
    FailedCreateModel = {"ru": "Не удалось создать модель. %s",
                         "eng": "Error when creating the model: %s"}
    FailedDeleteModel = {"ru": "Не удалось удалить модель. %s",
                         "eng": "Error when deleting the model: %s"}
    # Training
    FailedStartTrain = {"ru": "Не удалось начать обучение. %s",
                        "eng": "Error when start training. %s"}
    FailedStopTrain = {"ru": "Ошибка при попытке остановить обучение: %s",
                       "eng": "Error when stop training model: %s"}
    FailedCleanTrain = {"ru": "Не удалось очистить обучение. %s",
                        "eng": "Error when clean training model: %s"}
    FailedSetInteractiveConfig = {"ru": "Не удалось обновить параметры обучения. %s",
                                  "eng": "Error when setting interactive config: %s"}
    FailedGetTrainingProgress = {"ru": "Не удалось получить прогресс обучения. %s",
                                 "eng": "Could not get the progress of the training progress: %s"}
    # Deploy
    FailedUploadDeploy = {"ru": "Не удалось загрузить деплой. %s",
                          "eng": "Error when uploading deploy: %s"}
    FailedGetUploadDeployResult = {"ru": "Не удалось получить прогресс загрузки деплоя. %s",
                                   "eng": "Error when getting upload deploy result: %s"}


# Base Exceptions


class ExchangeBaseException(Exception):
    class Meta:
        message: ExceptionMessages = ExceptionMessages.UnknownError

    def __init__(self, *args, **kwargs):
        if not args:
            args = (self.Meta.message.value.get(kwargs.get('lang', 'ru')),)
        super().__init__(*args)


class ValueException(ExchangeBaseException):
    def __init__(self, __value: Any, lang: str = 'ru'):
        super().__init__(self.Meta.message.value.get(lang) % str(__value))


# Agent


class FileNotFoundException(ValueException):
    class Meta:
        message = ExceptionMessages.FileNotFound


class CallMethodNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.CallMethodNotFound

    def __init__(self, __class: Any, __method: str, lang: str = 'ru'):
        super().__init__(self.Meta.message.value.get(lang) % (str(__class), str(__method)))


class MethodNotCallableException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.MethodNotCallable

    def __init__(self, __class: Any, __method: str, lang: str = 'ru'):
        super().__init__(self.Meta.message.value.get(lang) % (str(__method), str(__class)))


# Project exceptions


class FailedGetProjectsInfoException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetProjectsInfo


class FailedSaveProjectException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedSaveProject


class FailedLoadProjectException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedLoadProject


class ProjectNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.ProjectNotFound

    def __init__(self, __project: str, __target: Path, lang: str = 'ru'):
        super().__init__(self.Meta.message.value.get(lang) % ((str(__project)), str(__target)))


class ProjectAlreadyExistsException(ValueException):
    class Meta:
        message = ExceptionMessages.ProjectAlreadyExists


# Dataset exceptions


class FailedChoiceDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedChoiceDataset


class FailedDeleteDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedDeleteDataset


class FailedGetProgressDatasetChoiceException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetProgressDatasetChoice


class FailedGetDatasetsInfoException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetDatasetsInfo


class FailedLoadDatasetsSourceException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedLoadDatasetsSource


class FailedLoadProgressDatasetsSource(ValueException):
    class Meta:
        message = ExceptionMessages.FailedLoadProgressDatasetsSource


class FailedGetDatasetsSourcesException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetDatasetsSources


class DatasetCanNotBeDeletedException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.DatasetCanNotBeDeleted

    def __init__(self, __dataset: str, __group: str, lang: str = 'ru'):
        super().__init__(self.Meta.message.value.get(lang) % ((str(__dataset)), str(__group)))


class FailedCreateDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedCreateDataset


# Modeling exceptions


class ModelAlreadyExistsException(ValueException):
    class Meta:
        message = ExceptionMessages.ModelAlreadyExists


class FailedGetModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetModel


class FailedValidateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedValidateModel


class FailedUpdateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedUpdateModel


class FailedGetModelsListException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetModelsList


class FailedCreateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedCreateModel


class FailedDeleteModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedDeleteModel


# Training exceptions


class FailedStopTrainException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedStopTrain


class FailedCleanTrainException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedCleanTrain


class FailedGetTrainingProgressException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetTrainingProgress


class FailedSetInteractiveConfigException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedSetInteractiveConfig


# Deploy exceptions


class FailedUploadDeployException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedUploadDeploy


class FailedGetUploadDeployResultException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetUploadDeployResult
