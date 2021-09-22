from pathlib import Path
from rest_framework import serializers

from .validators import validate_filepath


class FilePathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validate_filepath] + kwargs.get("validators", []),
            }
        )
        super().__init__(**kwargs)

    def to_representation(self, value) -> Path:
        value = super().to_representation(value)
        return Path(value)
