from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from apps.api.validators import validate_slug
from apps.plugins.project import project


class ReloadSerializer(serializers.ListSerializer):
     child=serializers.IntegerField(min_value=0)

    # def validate(self, indexes):
    #     if len(indexes) and project.deploy.data:
    #         _max = max(indexes)
    #         _len = len(project.deploy.data)
    #         if _max >= _len:
    #             raise ValidationError(
    #                 {
    #                     "indexes": f"Максимальный индекс списка `{_len-1}`, получено `{_max}`"
    #                 }
    #             )
    #     return super().validate(indexes)


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
