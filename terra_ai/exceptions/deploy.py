from enum import Enum

from .base import TerraBaseException


class DeployMessages(str, Enum):
    RequestAPI = "Error API request to deploy server"


class DeployException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of deploy"


class RequestAPIException(DeployException):
    class Meta:
        message: str = DeployMessages.RequestAPI.value
