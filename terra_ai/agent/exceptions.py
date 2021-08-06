from typing import Any
from enum import Enum


class ExceptionMessages(str, Enum):
    CallMethodNotFound = "Instance of `%s` must have method `%s`"
    MethodNotCallable = "Method `%s` of instance of `%s` must be callable"


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
