from typing import Any
from enum import Enum


class ExceptionMessages(str, Enum):
    NotOneArgument = (
        "Instance of %s must call with 1 argument but received %s arguments"
    )
    CallMethodNotFound = "Instance of %s must have method %s"
    MethodNotCallable = "Method %s of instance of %s must be callable"


class ExchangeBaseException(Exception):
    pass


class NotOneArgumentException(ExchangeBaseException):
    def __init__(self, __class: Any, __len: int):
        super().__init__(ExceptionMessages.NotOneArgument % (str(__class), str(__len)))


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
