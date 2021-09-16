from pathlib import Path
from typing import Any
from enum import Enum


class ExceptionMessages(str, Enum):
    # Agent
    UnknownError = "Unknown error"
    CallMethodNotFound = "Instance of `%s` must have method `%s`"
    MethodNotCallable = "Method `%s` of instance of `%s` must be callable"
    ModelAlreadyExists = "Model `%s` already exists"
    FileNotFound = "No such file or directory: %s"
    # Project
    ProjectAlreadyExists = "Project `%s` already exists"
    ProjectNotFound = "Project `%s` not found in `%s`"
    FailedGetProjectsInfo = "Error when getting projects info: %s"
    FailedSaveProject = "Error when saving project: %s"
    FailedLoadProject = "Error when loading project: %s"
    # Dataset
    FailedChoiceDataset = "Error when choosing dataset: %s"
    FailedDeleteDataset = "Dataset could not be deleted: %s"
    DatasetCanNotBeDeleted = "Dataset `%s` from group `%s` can't be deleted"
    FailedGetProgressDatasetChoice = (
        "Could not get the progress of the dataset choice: %s"
    )
    FailedGetDatasetsInfo = "Error when getting datasets info: %s"
    FailedLoadDatasetsSource = "Error when loading datasets sour: %s"
    FailedLoadProgressDatasetsSource = (
        "Error when loading progress of datasets info: %s"
    )
    FailedGetDatasetsSources = "Could not get the datasets info: %s"
    FailedCreateDataset = "Ошибка создания датасета: %s"
    # Modeling
    FailedGetModelsList = "Could not get the models list: %s"
    FailedGetModel = "Error when getting the model: %s"
    FailedValidateModel = "Error when validating the model: %s"
    FailedUpdateModel = "Error when updating the model: %s"
    FailedCreateModel = "Error when creating the model: %s"
    FailedDeleteModel = "Error when deleting the model: %s"
    # Training
    FailedStartTrain = "Error when start training model: %s"
    FailedStopTrain = "Error when stop training model: %s"
    FailedCleanTrain = "Error when clean training model: %s"
    FailedSetInteractiveConfig = "Error when setting interactive config: %s"
    FailedGetTrainingProgress = (
        "Could not get the progress of the training progress: %s"
    )
    # Deploy
    FailedUploadDeploy = "Error when uploading deploy: %s"
    FailedGetUploadDeployResult = "Error when getting upload deploy result: %s"


# Base Exceptions


class ExchangeBaseException(Exception):
    class Meta:
        message: ExceptionMessages = ExceptionMessages.UnknownError

    def __init__(self, *args, **kwargs):
        if not args:
            args = (self.Meta.message.value,)
        super().__init__(*args)


class ValueException(ExchangeBaseException):
    def __init__(self, __value: Any):
        super().__init__(self.Meta.message.value % str(__value))


# Agent


class FileNotFoundException(ValueException):
    class Meta:
        message = ExceptionMessages.FileNotFound


class CallMethodNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.CallMethodNotFound

    def __init__(self, __class: Any, __method: str):
        super().__init__(self.Meta.message.value % (str(__class), str(__method)))


class MethodNotCallableException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.MethodNotCallable

    def __init__(self, __class: Any, __method: str):
        super().__init__(self.Meta.message.value % (str(__method), str(__class)))


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

    def __init__(self, __project: str, __target: Path):
        super().__init__(self.Meta.message.value % ((str(__project)), str(__target)))


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

    def __init__(self, __dataset: str, __group: str):
        super().__init__(self.Meta.message.value % ((str(__dataset)), str(__group)))


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
