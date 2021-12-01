import requests

from django.conf import settings

from apps.api.base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)

from .serializers import CreateSerializer


class ListAPIView(BaseAPIView):
    def post(self, request):
        return BaseResponseSuccess(settings.USER_SERVERS)


class CreateAPIView(BaseAPIView):
    def post(self, request):
        serializer = CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        response = requests.post(
            f"{settings.TERRA_API_URL}/server/create/",
            json={"config": settings.USER_PORT, **serializer.validated_data},
        ).json()
        if not response.get("success"):
            return BaseResponseErrorGeneral(
                f'Не удалось создать конфигурацию сервера: {response.get("error")}'
            )
        return BaseResponseSuccess(response.get("data"))
