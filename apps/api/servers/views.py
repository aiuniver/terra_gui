import json
import requests

from typing import Dict, Any

from django.conf import settings
from rest_framework.exceptions import APIException

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.servers.serializers import (
    CreateSerializer,
    ServerSerializer,
    ServerData,
    ServerFullData,
)


def remote_request(url: str, data: dict = None) -> Any:
    if data is None:
        data = {}
    response = requests.post(
        f"{settings.TERRA_API_URL}{url}",
        json={"config": settings.USER_PORT, **data},
    )
    if response.ok:
        data = response.json()
        if data.get("success"):
            return data.get("data")
        else:
            raise APIException(data.get("error"))
    else:
        response.raise_for_status()


class ServersListMixinAPIView(BaseAPIView):
    def get_servers(self, state: str = None) -> Dict[int, dict]:
        servers = []
        for server in remote_request(
            f'/server/list/{f"?state={state}" if state else ""}'
        ):
            server["state"] = {"name": server.get("state")}
            servers.append(json.loads(ServerData(**server).json(ensure_ascii=False)))
        return servers


class ListAPIView(ServersListMixinAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(self.get_servers())


class CreateAPIView(ServersListMixinAPIView):
    @decorators.serialize_data(CreateSerializer)
    def post(self, request, serializer, **kwargs):
        server = remote_request("/server/create/", serializer.validated_data)
        server["state"] = {"name": server.get("state")}
        return BaseResponseSuccess(
            json.loads(ServerData(**server).json(ensure_ascii=False))
        )


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        server = remote_request("/server/get/", serializer.validated_data)
        server["state"] = {"name": server.get("state")}
        return BaseResponseSuccess(
            json.loads(ServerFullData(**server).json(ensure_ascii=False))
        )


class SetupAPIView(ServersListMixinAPIView):
    @decorators.serialize_data(ServerSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            remote_request("/server/setup/", serializer.validated_data)
        )


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
                    self.get_servers("ready"),
                )
            )
        )
