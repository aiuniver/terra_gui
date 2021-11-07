from rest_framework import serializers


class SaveSerializer(serializers.Serializer):
    first_name = serializers.CharField()
    last_name = serializers.CharField()
