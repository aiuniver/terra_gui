import tensorflow
import pydantic

from . import tensor_flow as tf_exceptions
from .data import PydanticException

from .base import TerraBaseException


def terra_exception(exception: Exception) -> TerraBaseException:
    """Принимает на вход любой тип исключения Exception и возвращает исключение, унаследованное от TerraBaseException"""

    if not isinstance(exception, Exception):
        raise TypeError(
            f"Функция ожидала на вход объект исключения, но получила '{type(exception).__name__}'"
        )

    if isinstance(
        exception, pydantic.ValidationError
    ):  # нативные исключения от Pydantic
        raise PydanticException(exception)

    if isinstance(
        exception, tensorflow.errors.OpError
    ):  # нативные исключения от TensorFlow
        return getattr(
            tf_exceptions, exception.__class__.__name__, tf_exceptions.UnknownError
        )(exception.message)

    if isinstance(exception, TerraBaseException):  # исключения от TerraAI
        return exception

    return TerraBaseException(str(exception))  # неопределенные исключения
