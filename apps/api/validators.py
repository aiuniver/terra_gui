import re

from django.core.validators import RegexValidator


validate_slug = RegexValidator(
    re.compile(r"^[a-z]+[a-z0-9\-_]*$"),
    message="Разрешены только латинские символы в нижнем регистре, цифры и символы '-' и '_'",
)
