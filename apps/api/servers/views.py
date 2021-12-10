import requests

from django.conf import settings

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.servers.serializers import CreateSerializer


class ListAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(settings.USER_SERVERS)


class CreateAPIView(BaseAPIView):
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
        return BaseResponseSuccess(response_data.get("data"))
