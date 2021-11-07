from rest_framework import serializers


class SaveSerializer(serializers.Serializer):
    name = serializers.CharField()
    overwrite = serializers.BooleanField(default=False)
