from pathlib import Path
from typing import Any
from enum import Enum


class ExceptionMessages(str, Enum):
    UnknownError = "Unknown error"
    CallMethodNotFound = "Instance of `%s` must have method `%s`"
    MethodNotCallable = "Method `%s` of instance of `%s` must be callable"
    ModelAlreadyExists = "Model `%s` already exists"
    ProjectAlreadyExists = "Project `%s` already exists"
    ProjectNotFound = "Project `%s` not found in `%s`"
    FileNotFound = "No such file or directory: %s"
    DatasetCanNotBeDeleted = "Dataset `%s` from group `%s` can't be deleted"
    FailedGetModel = "Error when getting the model: %s"
    FailedValidateModel = "Error when validating the model: %s"
    FailedUpdateModel = "Error when updating the model: %s"
    FailedCreateModel = "Error when creating the model: %s"
    FailedDeleteModel = "Error when deleting the model: %s"
    FailedStartTrain = "Error when start training model: %s"
    FailedSetInteractiveConfig = "Error when setting interactive config: %s"
    FailedUploadDeploy = "Error when uploading deploy: %s"
    FailedGetUploadDeployResult = "Error when getting upload deploy result: %s"
    FailedCreateDataset = "Ошибка создания датасета: %s"
    FailedGetProjectsInfo = "Error when getting projects info: %s"
    FailedSaveProject = "Error when saving project: %s"
    FailedLoadProject = "Error when loading project: %s"
    FailedChoiceDataset = "Error when choosing dataset: %s"
    FailedDeleteDataset = "Dataset could not be deleted: %s"

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


class ProjectNotFoundException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.ProjectNotFound

    def __init__(self, __project: str, __target: Path):
        super().__init__(self.Meta.message.value % ((str(__project)), str(__target)))


class DatasetCanNotBeDeletedException(ExchangeBaseException):
    class Meta:
        message = ExceptionMessages.DatasetCanNotBeDeleted

    def __init__(self, __dataset: str, __group: str):
        super().__init__(self.Meta.message.value % ((str(__dataset)), str(__group)))


class ModelAlreadyExistsException(ValueException):
    class Meta:
        message = ExceptionMessages.ModelAlreadyExists


class ProjectAlreadyExistsException(ValueException):
    class Meta:
        message = ExceptionMessages.ProjectAlreadyExists


class FailedGetModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetModel


class FailedValidateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedValidateModel


class FailedUpdateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedUpdateModel


class FailedCreateModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedCreateModel


class FailedDeleteModelException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedDeleteModel


class FailedStartTrainException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedStartTrain


class FailedSetInteractiveConfigException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedSetInteractiveConfig


class FailedUploadDeployException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedUploadDeploy


class FailedGetUploadDeployResultException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetUploadDeployResult


class FailedCreateDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedCreateDataset


class FailedGetProjectsInfoException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedGetProjectsInfo


class FileNotFoundException(ValueException):
    class Meta:
        message = ExceptionMessages.FileNotFound


class FailedSaveProjectException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedSaveProject


class FailedLoadProjectException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedLoadProject


class FailedChoiceDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedChoiceDataset


class FailedDeleteDatasetException(ValueException):
    class Meta:
        message = ExceptionMessages.FailedDeleteDataset