import re

from django.core.exceptions import ValidationError


def validate_restriction_name(value):
    value_match = re.match("^[a-zA-Zа-яА-Я0-9\s_\-]+$", value)
    if not value_match:
        raise ValidationError(
            "Можно использовать только латиницу, кириллицу, цифры, пробел и символы `-_`"
        )
