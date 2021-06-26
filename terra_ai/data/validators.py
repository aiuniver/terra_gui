"""
## Общие валидаторы

```
class SomeModel:
    length: Optional[int]

    _validate_positive_integer = validator("length", allow_reuse=True)(
        validate_positive_integer
    )
```
"""

import re

from typing import Optional

from .exceptions import AliasException, PositiveIntegerException, PartValueException


def validate_alias(value: str) -> Optional[str]:
    """
    Используется регулярное выражение `^[a-z]+[a-z0-9_]*$`
    """
    if not re.match("^[a-z]+[a-z0-9_]*$", value):
        raise AliasException(value)
    return value


def validate_positive_integer(value: int) -> Optional[int]:
    """
    Используется для проверки натурального числа в случае, если значение не `None`
    """
    if value is not None and value < 1:
        raise PositiveIntegerException(value)
    return value


def validate_part_value(value: float) -> Optional[float]:
    """
    Валидация доли: значение должно быть `float` между `0` и `1`
    """
    if value is not None and (value < 0 or value > 1):
        raise PartValueException(value)
    return value
