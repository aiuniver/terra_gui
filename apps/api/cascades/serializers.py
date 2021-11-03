from rest_framework import serializers


class CascadeGetSerializer(serializers.Serializer):
    value = serializers.CharField()
