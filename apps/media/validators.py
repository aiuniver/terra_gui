import os

from pathlib import Path
from typing import Optional
from rest_framework.exceptions import ValidationError

from terra_ai import settings as terra_ai_settings


def validate_filepath(value) -> Optional[Path]:
    if value is None:
        return value
    value = Path(value)
    if not value.is_file():
        raise ValidationError(f"Файл {value} не существует")
    return value


def validate_project_path(value) -> Optional[Path]:
    if value is None:
        return value
    value = Path(value)
    if (
        not str(value).startswith(f"{terra_ai_settings.PROJECT_PATH.base}{os.sep}")
        and not str(value).startswith(f"{terra_ai_settings.CASCADE_PATH}{os.sep}")
        and not str(value).startswith(f"{terra_ai_settings.DEPLOY_PATH}{os.sep}")
    ):
        raise ValidationError(f"Файл {value} запрещен для вывода")
    return value
