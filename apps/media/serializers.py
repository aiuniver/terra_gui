from rest_framework import serializers


class RequestFileDataSerializer(serializers.Serializer):
    hash = serializers.CharField()
