from rest_framework import serializers


class VersionsSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class DeleteSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()
