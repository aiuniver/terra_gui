import json
import requests

from typing import Dict

from django.conf import settings

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.servers.serializers import (
    CreateSerializer,
    ServerSerializer,
    ServerData,
    ServerFullData,
)


class ServersListMixinAPIView(BaseAPIView):
    def get_servers(self) -> Dict[int, dict]:
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/servers/",
            json={"config": settings.USER_PORT},
        ).json()
        servers_data = response_data.get("data")
        servers = []
        for server in servers_data:
            server["state"] = {
                "name": server.get("state"),
                "error": server.get("error"),
            }
            servers.append(json.loads(ServerData(**server).json(ensure_ascii=False)))
        return servers

    def get_servers_ready(self) -> Dict[int, dict]:
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/servers/ready/",
            json={"config": settings.USER_PORT},
        ).json()
        servers_data = response_data.get("data")
        servers = []
        for server in servers_data:
            server["state"] = {
                "name": server.get("state"),
                "error": server.get("error"),
            }
            servers.append(
                json.loads(ServerFullData(**server).json(ensure_ascii=False))
            )
        return servers


class ListAPIView(ServersListMixinAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(self.get_servers())


class CreateAPIView(ServersListMixinAPIView):
    @decorators.serialize_data(CreateSerializer)
    def post(self, request, serializer, **kwargs):
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/server/create/",
            json={"config": settings.USER_PORT, **serializer.validated_data},
        ).json()
        if not response_data.get("success"):
            raise ValueError(
                f'Не удалось создать конфигурацию сервера: {response_data.get("error")}'
            )
        return BaseResponseSuccess(
            {
                "id": response_data.get("data").get("id"),
                "servers": self.get_servers(),
            }
        )


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/server/",
            json={"config": settings.USER_PORT, **serializer.validated_data},
        ).json()
        server = response_data.get("data")
        server["state"] = {
            "name": server.get("state"),
            "error": server.get("error"),
        }
        return BaseResponseSuccess(
            json.loads(ServerFullData(**server).json(ensure_ascii=False))
        )


class SetupAPIView(ServersListMixinAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/server/setup/",
            json={"config": settings.USER_PORT, **serializer.validated_data},
        ).json()
        return BaseResponseSuccess(self.get_servers())


class ReadyAPIView(ServersListMixinAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            [{"label": "Демо-панель TerraAI", "value": 0}]
            + list(
                map(
                    lambda server: {
                        "label": f'{server.get("domain_name")} [{server.get("ip_address")}]',
                        "value": server.get("id"),
                    },
                    self.get_servers_ready(),
                )
            )
        )
