from rest_framework.exceptions import ValidationError

from apps.plugins.project import project


def validate_reload_id(value: str) -> str:
    if not project.deploy.data.get(value):
        raise ValidationError(f"Не существует пресета с ID `{str(value)}`")
    return value
