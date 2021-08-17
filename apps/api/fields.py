from rest_framework import serializers

from .validators import validate_directory_path


class DirectoryPathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validate_directory_path],
            }
        )
        super().__init__(**kwargs)
