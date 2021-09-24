from enum import Enum

from .base import TerraBaseException


class DeployMessages(str, Enum):
    RequestAPI = "Error API request to deploy server"
    Rsync = "Rsync error: %s"
    MethodNotImplemented = "Method `%s` must be implemented in class `%s`"


class DeployException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of deploy"


class RequestAPIException(DeployException):
    class Meta:
        message: str = DeployMessages.RequestAPI.value


class RsyncException(DeployException):
    class Meta:
        message: str = DeployMessages.Rsync

    def __init__(self, __error: str, *args):
        super().__init__(self.Meta.message % str(__error), *args)


class MethodNotImplementedException(DeployException):
    class Meta:
        message: str = DeployMessages.MethodNotImplemented

    def __init__(self, __method: str, __class: str, *args):
        super().__init__(self.Meta.message % (str(__method), str(__class)), *args)
