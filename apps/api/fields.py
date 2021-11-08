from rest_framework import serializers

from . import validators


class DirectoryPathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validators.validate_directory_path],
            }
        )
        super().__init__(**kwargs)


class DirectoryOrFilePathField(serializers.CharField):
    def __init__(self, **kwargs):
        kwargs.update(
            {
                "validators": [validators.validate_directory_or_file_path],
            }
        )
        super().__init__(**kwargs)
