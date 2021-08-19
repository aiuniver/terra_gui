from enum import Enum

from .base import TerraBaseException


class DeployMessages(str, Enum):
    RequestAPI = "Error API request to deploy server"
    Rsync = "Rsync error: %s"


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
