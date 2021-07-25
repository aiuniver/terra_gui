from rest_framework import serializers


class ChoiceSerializer(serializers.Serializer):
    group = serializers.CharField(required=True)
    alias = serializers.CharField(required=True)


class SourceLoadSerializer(serializers.Serializer):
    mode = serializers.CharField(required=True)
    value = serializers.CharField(required=True)
