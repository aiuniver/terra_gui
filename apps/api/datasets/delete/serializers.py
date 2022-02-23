from rest_framework import serializers


class DeleteSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class DeleteVersionSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()
    version = serializers.CharField()
