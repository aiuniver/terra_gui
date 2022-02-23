from rest_framework import serializers


class LoadSerializer(serializers.Serializer):
    mode = serializers.CharField()
    value = serializers.CharField()
    name = serializers.CharField()
    tags = serializers.ListSerializer(child=serializers.CharField(), default=[])
    architecture = serializers.CharField()
