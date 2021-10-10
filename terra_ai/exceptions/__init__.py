import tensorflow

from . import tensor_flow as tf_exceptions

from .base import TerraBaseException


def terra_exception(exception: Exception) -> TerraBaseException:
    """Принимает на вход любой тип исключения Exception и возвращает исключение, унаследованное от TerraBaseException"""

    if not isinstance(exception, Exception):
        raise TypeError(f"Функция ожидала на вход объект исключения, но получила '{type(exception).__name__}'")

    if isinstance(exception, tensorflow.errors.OpError):
        return getattr(
            tf_exceptions, exception.__class__.__name__, tf_exceptions.UnknownError
        )(exception.message)

    return TerraBaseException(str(exception))
