from rest_framework import serializers


class ChoiceSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()
    version = serializers.CharField()
    reset_model = serializers.BooleanField(default=False)
