from rest_framework import serializers

from apps.api.datasets import serializers as datasets_serializers
from apps.api.modeling import serializers as modeling_serializers


class ValidateDatasetModelSerializer(serializers.Serializer):
    dataset = datasets_serializers.ChoiceSerializer(required=False)
    model = modeling_serializers.ModelGetSerializer(required=False)

    def validate(self, attrs):
        _dataset = attrs.get("dataset")
        _model = attrs.get("model")
        if (not _dataset and not _model) or (_dataset and _model):
            raise serializers.ValidationError(
                "Необходимо указать либо датасет, либо модель"
            )
        return super().validate(attrs)
