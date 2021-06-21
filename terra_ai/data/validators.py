import re

from typing import Optional


def validate_alias(value: str) -> Optional[str]:
    if not re.match("^[a-z]+[a-z0-9_]*$", value):
        raise ValueError(
            f'{value}: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
        )
    return value


def validate_positive_integer(value: int) -> Optional[int]:
    if value is not None and value < 1:
        raise ValueError(f"{value}: Value must be greater or equivalent then 1")
    return value
