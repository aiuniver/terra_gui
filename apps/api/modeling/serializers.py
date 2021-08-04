from rest_framework import serializers


class ModelGetSerializer(serializers.Serializer):
    value = serializers.CharField(required=True)
