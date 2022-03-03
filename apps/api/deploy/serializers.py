import json

from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from apps.api import remote
from apps.api.validators import validate_slug
from apps.plugins.frontend.choices import DeployTypePageChoice
from apps.api.servers.serializers import ServerDeployData

from terra_ai.settings import DEPLOY_PRESET_COUNT


class GetSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=DeployTypePageChoice.items_tuple())
    name = serializers.CharField()


class ReloadSerializer(serializers.ListSerializer):
    child = serializers.IntegerField(min_value=0, max_value=DEPLOY_PRESET_COUNT - 1)


class UploadSerializer(serializers.Serializer):
    deploy = serializers.CharField(validators=[validate_slug])
    replace = serializers.BooleanField(default=False)
    use_sec = serializers.BooleanField(default=False)
    sec = serializers.CharField(required=False)
    server = serializers.IntegerField(min_value=0)

    def validate_server(self, value: int):
        if value == 0:
            with open("./rsa.key") as env_file_ref:
                data = {
                    "id": 0,
                    "state": {"name": "ready"},
                    "domain_name": "srv1.demo.neural-university.ru",
                    "ip_address": "188.124.47.137",
                    "user": "terra",
                    "private_ssh_key": env_file_ref.read(),
                }
        else:
            data = remote.request("/server/get/", data={"id": value})
            data.update({"state": {"name": data.get("state", "idle")}})
        if data.get("state", {}).get("name", "idle") != "ready":
            raise ValidationError("Сервер не готов к работе")
        return json.loads(ServerDeployData(**data).json(ensure_ascii=False))

    def validate(self, attrs):
        if attrs.get("use_sec"):
            if not attrs.get("sec"):
                raise ValidationError({"sec": "Введите пароль"})
        else:
            attrs.update({"sec": ""})
        return super().validate(attrs)
