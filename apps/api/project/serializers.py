from rest_framework import serializers


class NameSerializer(serializers.Serializer):
    name = serializers.CharField()


class SaveSerializer(serializers.Serializer):
    name = serializers.CharField()
    overwrite = serializers.BooleanField(default=False)
