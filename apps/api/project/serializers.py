from rest_framework import serializers


class NameSerializer(serializers.Serializer):
    name = serializers.CharField(required=True)
