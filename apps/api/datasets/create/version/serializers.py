from rest_framework import serializers

from terra_ai.data.datasets.extra import DatasetGroupChoice


class VersionSerializer(serializers.Serializer):
    group = serializers.ChoiceField(choices=DatasetGroupChoice.values())
    alias = serializers.CharField()
    version = serializers.CharField(required=False, allow_blank=True)
