from rest_framework import serializers


class ModelLoadSerializer(serializers.Serializer):
    value = serializers.CharField(required=True)
