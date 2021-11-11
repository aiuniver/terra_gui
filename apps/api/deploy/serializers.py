from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from terra_ai.settings import DEPLOY_PRESET_COUNT

from apps.api.validators import validate_slug


class GetSerializer(serializers.Serializer):
    type = serializers.CharField()
    value = serializers.CharField()

class ReloadSerializer(serializers.ListSerializer):
    child = serializers.IntegerField(min_value=0, max_value=DEPLOY_PRESET_COUNT - 1)


class UploadSerializer(serializers.Serializer):
    deploy = serializers.CharField(validators=[validate_slug])
    replace = serializers.BooleanField(default=False)
    use_sec = serializers.BooleanField(default=False)
    sec = serializers.CharField(required=False)

    def validate(self, attrs):
        if attrs.get("use_sec"):
            if not attrs.get("sec"):
                raise ValidationError({"sec": "Введите пароль"})
        else:
            attrs.update({"sec": ""})
        return super().validate(attrs)
