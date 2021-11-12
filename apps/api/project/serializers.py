from rest_framework import serializers


class NameSerializer(serializers.Serializer):
    name = serializers.CharField()


class SaveSerializer(serializers.Serializer):
    name = serializers.CharField()
    overwrite = serializers.BooleanField(default=False)


class LoadSerializer(serializers.Serializer):
    value = serializers.CharField()


class DeleteSerializer(serializers.Serializer):
    path = serializers.CharField()
