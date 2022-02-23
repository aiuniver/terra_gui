from rest_framework import serializers

from terra_ai.data.datasets.extra import DatasetGroupChoice


class ChoiceSerializer(serializers.Serializer):
    group = serializers.ChoiceField(choices=DatasetGroupChoice.values())
    alias = serializers.CharField()
    version = serializers.CharField()
    reset_model = serializers.BooleanField(default=False)
