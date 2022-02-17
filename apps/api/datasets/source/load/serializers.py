from rest_framework import serializers


class LoadSerializer(serializers.Serializer):
    mode = serializers.CharField()
    value = serializers.CharField()
    architecture = serializers.CharField()
