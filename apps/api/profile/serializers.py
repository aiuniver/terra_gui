from rest_framework import serializers


class SaveSerializer(serializers.Serializer):
    login = serializers.CharField(required=True)
    first_name = serializers.CharField(required=True)
    last_name = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    token = serializers.CharField(required=True)
