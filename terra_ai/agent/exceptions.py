from typing import Any
from enum import Enum


class ExceptionMessages(str, Enum):
    CallMethodNotFound = "Instance of `%s` must have method `%s`"
    MethodNotCallable = "Method `%s` of instance of `%s` must be callable"
    ModelAlreadyExists = "Model `%s` already exists"
    FailedGetModel = "Error when getting the model: %s"
    FailedValidateModel = "Error when validating the model: %s"
    FailedUpdateModel = "Error when updating the model: %s"
    FailedCreateModel = "Error when creating the model: %s"
    FailedDeleteModel = "Error when deleting the model: %s"
    FailedStartTrain = "Error when start training model: %s"
    FailedSetInteractiveConfig = "Error when setting interactive config: %s"
    FailedUploadDeploy = "Error when uploading deploy: %s"
    FailedGetUploadDeployResult = "Error when getting upload deploy result: %s"


class ExchangeBaseException(Exception):
    pass


class CallMethodNotFoundException(ExchangeBaseException):
    def __init__(self, __class: Any, __method: str):
        super().__init__(
            ExceptionMessages.CallMethodNotFound % (str(__class), str(__method))
        )


class MethodNotCallableException(ExchangeBaseException):
    def __init__(self, __class: Any, __method: str):
        super().__init__(
            ExceptionMessages.MethodNotCallable % (str(__method), str(__class))
        )


class ModelAlreadyExistsException(ExchangeBaseException):
    def __init__(self, __name: str):
        super().__init__(ExceptionMessages.ModelAlreadyExists % str(__name))


class FailedGetModelException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedGetModel % __error)


class FailedValidateModelException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedValidateModel % __error)


class FailedUpdateModelException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedUpdateModel % __error)


class FailedCreateModelException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedCreateModel % __error)


class FailedDeleteModelException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedDeleteModel % __error)


class FailedStartTrainException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedStartTrain % __error)


class FailedSetInteractiveConfigException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedSetInteractiveConfig % __error)


class FailedUploadDeployException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedUploadDeploy % __error)


class FailedGetUploadDeployResultException(ExchangeBaseException):
    def __init__(self, __error: str):
        super().__init__(ExceptionMessages.FailedGetUploadDeployResult % __error)


