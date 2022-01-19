from rest_framework import serializers


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    user_token = serializers.CharField()
    is_colab = serializers.BooleanField(default=False)
