import os
import re

from django.core.validators import RegexValidator, ValidationError


validate_slug = RegexValidator(
    re.compile(r"^[a-z]+[a-z0-9_]*$"),
    message="Разрешены только латинские символы в нижнем регистре, цифры и символ '_'",
)


def validate_directory_path(value: str) -> str:
    if not value:
        return value
    if not os.path.isdir(value):
        raise ValidationError("Неверный путь к директории")
    return value
