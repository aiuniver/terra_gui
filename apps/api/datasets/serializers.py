from rest_framework import serializers


class SourceLoadSerializer(serializers.Serializer):
    mode = serializers.CharField(required=True)
    value = serializers.CharField(required=True)
