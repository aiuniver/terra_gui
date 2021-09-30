from rest_framework import serializers

from .validators import validate_directory_path, validate_directory_or_file_path


class DirectoryPathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validate_directory_path],
            }
        )
        super().__init__(**kwargs)


class DirectoryOrFilePathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validate_directory_or_file_path],
            }
        )
        super().__init__(**kwargs)
