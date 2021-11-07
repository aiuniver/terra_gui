from pathlib import Path
from typing import Any
from enum import Enum

from terra_ai.settings import LANGUAGE


class ExceptionMessages(dict, Enum):
    # Agent
    UnknownError = {"ru": "%s", "eng": "%s"}
    CallMethodNotFound = {"ru": "Экземпляр класса `%s` не имеет метод `%s`",
                          "eng": "Instance of `%s` must have method `%s`"}
    MethodNotCallable = {"ru": "Метод `%s` у экземпляра класса `%s` должен быть вызываемым",
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
                         "eng": "Error when loading project. %s"}
    # Dataset
    FailedChoiceDataset = {"ru": "Не удалось выбрать датасет. %s",
                           "eng": "Error when choosing dataset: %s"}
    FailedDeleteDataset = {"ru": "Не удалось удалить датасет. %s",
                           "eng": "Dataset could not be deleted: %s"}
    DatasetCanNotBeDeleted = {"ru": "Датасет `%s` из группы `%s` нельзя удалить",
                              "eng": "The dataset `%s` cannot be deleted from the group `%s`"}
    FailedGetProgressDatasetChoice = {"ru": "Не удалось получить прогресс выбора датасета. %s",
                                      "eng": "Could not get the progress of the dataset choice: %s"}
    FailedGetDatasetsInfo = {"ru": "Не удалось получить информацию о датасетах. %s",
                             "eng": "Error when getting datasets info: %s"}
    FailedLoadDatasetsSource = {"ru": "Не удалось загрузить исходники датасета. %s",
                                "eng": "Error when loading datasets source: %s"}
    FailedLoadProgressDatasetsSource = {"ru": "Не удалось получить информацию о прогрессе выбора датасета. %s",
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
    FailedDeleteModel = {"ru": "Не получилось удалить модель. %s",
                         "eng": "Error when deleting the model: %s"}
    # Training
    FailedStartTrain = {"ru": "Ошибка при старте обучения: %s",
                        "eng": "Error when start training. %s"}
    FailedStopTrain = {"ru": "Ошибка при остановке обучения: %s",
                       "eng": "Error when stop training model: %s"}
    FailedCleanTrain = {"ru": "Ошибка при сбросе обучения: %s",
                        "eng": "Error when clean training model: %s"}
    FailedSetInteractiveConfig = {"ru": "Не удалось обновить параметры обучения. %s",
                                  "eng": "Error when setting interactive config: %s"}
    FailedGetTrainingProgress = {"ru": "Не удалось получить прогресс обучения. %s",
                                 "eng": "Could not get the progress of the training progress: %s"}
    FailedSaveTrain = {"ru": "Не удалось сохранить обучение. %s",
                       "eng": "Error when save training. %s"}

    # Deploy
    FailedGetDeployPresets = {"ru": "Не удалось получить данные пресетов деплоя. %s",
                              'eng': 'Error when getting deploy presets data. %s'}
    FailedGetDeployCollection = {'ru': 'Не удалось получить данные коллекции деплоя. %s',
                                 'eng': 'Error when getting deploy presets data. %s'}
    FailedUploadDeploy = {"ru": "Не удалось загрузить деплой. %s",
                          "eng": "Error when uploading deploy: %s"}
    FailedGetUploadDeployProgress = {"ru": "Не удалось получить прогресс загрузки деплоя. %s",
                                     "eng": "Error when getting upload deploy result: %s"}


# Base Exception

class ExchangeBaseException(Exception):
    class Meta:
        message: ExceptionMessages = ExceptionMessages.UnknownError

    def __init__(self, *args, **kwargs):
        error_msg = self.Meta.message.value.get(kwargs.get('lang', LANGUAGE))

        if args:
            error_msg = error_msg % args

        super().__init__(error_msg)


# Agent

class FileNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FileNotFound


class CallMethodNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.CallMethodNotFound

    def __init__(self, __class: Any, __method: str, **kwargs):
        super().__init__(str(__class), str(__method), **kwargs)


class MethodNotCallableException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.MethodNotCallable

    def __init__(self, __method: str, __class: Any, **kwargs):
        super().__init__(str(__method), str(__class), **kwargs)


# Project exceptions

class FailedGetProjectsInfoException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetProjectsInfo


class FailedSaveProjectException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedSaveProject


class FailedLoadProjectException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedLoadProject


class ProjectNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.ProjectNotFound

    def __init__(self, __project: str, __target: Path, **kwargs):
        super().__init__(str(__project), str(__target), **kwargs)


class ProjectAlreadyExistsException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.ProjectAlreadyExists


# Dataset exceptions

class FailedChoiceDatasetException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedChoiceDataset


class FailedDeleteDatasetException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedDeleteDataset


class FailedGetProgressDatasetChoiceException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetProgressDatasetChoice


class FailedGetDatasetsInfoException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetDatasetsInfo


class FailedLoadDatasetsSourceException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedLoadDatasetsSource


class FailedLoadProgressDatasetsSource(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedLoadProgressDatasetsSource


class FailedGetDatasetsSourcesException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetDatasetsSources


class DatasetCanNotBeDeletedException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.DatasetCanNotBeDeleted

    def __init__(self, __dataset: str, __group: str, **kwargs):
        super().__init__(str(__dataset), str(__group), **kwargs)


class FailedCreateDatasetException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedCreateDataset


# Modeling exceptions

class ModelAlreadyExistsException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.ModelAlreadyExists


class FailedGetModelException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetModel


class FailedValidateModelException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedValidateModel


class FailedUpdateModelException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedUpdateModel


class FailedGetModelsListException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetModelsList


class FailedCreateModelException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedCreateModel


class FailedDeleteModelException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedDeleteModel


# Training exceptions

class FailedStartTrainException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedStartTrain


class FailedStopTrainException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedStopTrain


class FailedCleanTrainException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedCleanTrain


class FailedGetTrainingProgressException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetTrainingProgress


class FailedSetInteractiveConfigException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedSetInteractiveConfig


class FailedSaveTrainException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedSaveTrain


# Deploy exceptions

class FailedGetDeployPresetsException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetDeployPresets


class FailedGetDeployCollectionException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetDeployCollection


class FailedUploadDeployException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedUploadDeploy


class FailedGetUploadDeployProgressException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.FailedGetUploadDeployProgress
