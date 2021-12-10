import requests

from typing import Dict

from django.conf import settings

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.servers.serializers import (
    CreateSerializer,
    InstructionSerializer,
    SetupSerializer,
)


class ServersListMixinAPIView(BaseAPIView):
    def get_servers(self) -> Dict[int, dict]:
        response_data = requests.post(
            f"{settings.TERRA_API_URL}/servers/",
            json={"config": settings.USER_PORT},
        ).json()
        return response_data.get("data")


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


class InstructionAPIView(BaseAPIView):
    @decorators.serialize_data(InstructionSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess()


class SetupAPIView(BaseAPIView):
    @decorators.serialize_data(SetupSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess()
