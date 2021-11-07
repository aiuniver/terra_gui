from pathlib import Path
from rest_framework.exceptions import ValidationError

from apps.plugins.project import project_path


def validate_filepath(value) -> Path:
    if value is None:
        return value
    value = Path(value)
    if not value.is_file():
        raise ValidationError(f"Файл {value} не существует")
    return value


def validate_project_path(value) -> Path:
    if value is None:
        return value
    value = Path(value)
    if not str(value).startswith(f"{project_path.base}/"):
        raise ValidationError(f"Файл {value} запрещен для вывода")
    return value
