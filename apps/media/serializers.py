from rest_framework import serializers

from .fields import FilePathField
from .validators import validate_project_path


class RequestFileDataSerializer(serializers.Serializer):
    path = FilePathField(validators=[validate_project_path])
