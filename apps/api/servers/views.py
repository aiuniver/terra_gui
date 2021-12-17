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


def remote_request(url: str, data: dict = None) -> requests.models.Response:
    if data is None:
        data = {}
    response = requests.post(
        f"{settings.TERRA_API_URL}{url}",
        json={"config": settings.USER_PORT, **data},
    )
    if response.ok:
        return response
    else:
        response.raise_for_status()


class ServersListMixinAPIView(BaseAPIView):
    def get_servers(self) -> Dict[int, dict]:
        response = remote_request("/server/list/").json()
        servers_data = response.get("data")
        servers = []
        for server in servers_data:
            server["state"] = {"name": server.get("state")}
            servers.append(json.loads(ServerData(**server).json(ensure_ascii=False)))
        return servers

    def get_servers_ready(self) -> Dict[int, dict]:
        response = remote_request("/server/ready/").json()
        servers_data = response.get("data")
        servers = []
        for server in servers_data:
            server["state"] = {"name": server.get("state")}
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
        response = remote_request("/server/create/", serializer.validated_data).json()
        if not response.get("success"):
            raise ValueError(
                f'Не удалось создать конфигурацию сервера: {response.get("error")}'
            )
        return BaseResponseSuccess(
            {
                "id": response.get("data").get("id"),
                "servers": self.get_servers(),
            }
        )


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        response = remote_request("/server/get/", serializer.validated_data).json()
        server = response.get("data")
        server["state"] = {"name": server.get("state")}
        return BaseResponseSuccess(
            json.loads(ServerFullData(**server).json(ensure_ascii=False))
        )


class SetupAPIView(ServersListMixinAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        remote_request("/server/setup/", serializer.validated_data)
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
