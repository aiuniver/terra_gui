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


def validate_alias(value: str) -> Optional[str]:
    """
    В основном используется в [`mixins.UniqueListMixin`](mixins.html#data.mixins.UniqueListMixin) для уникальной идентификации.
    Используется следующее регулярное выражение `^[a-z]+[a-z0-9_]*$`
    """
    if not re.match("^[a-z]+[a-z0-9_]*$", value):
        raise ValueError(
            f'{value}: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
        )
    return value


def validate_positive_integer(value: int) -> Optional[int]:
    """
    Используется для проверки натурального числа в случае, если значение не `None`
    """
    if value is not None and value < 1:
        raise ValueError(f"{value}: Value must be greater or equivalent then 1")
    return value


def validate_part_value(value: float) -> Optional[float]:
    """
    Валидация доли: значение должно быть `float` между `0` и `1`
    """
    if value is not None and (value < 0 or value > 1):
        raise ValueError(f"{value}: Value must be between 0 and 1")
    return value
