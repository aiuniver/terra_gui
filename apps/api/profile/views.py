import requests

from django.conf import settings

from apps.api.base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)

from . import serializers, utils


class SaveAPIView(BaseAPIView):
    def post(self, request):
        serializer = serializers.SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        data = dict(serializer.validated_data)
        data.update(
            {
                "email": settings.USER_EMAIL,
                "user_token": settings.USER_TOKEN,
            }
        )
        response = requests.post(
            f"{settings.TERRA_AI_EXCHANGE_API_URL}/update/", json=data
        )
        if not response.json().get("success"):
            return BaseResponseErrorGeneral("Не удалось обновить данные пользователя")
        utils.update_env_file(**serializer.validated_data)
        return BaseResponseSuccess()


class UpdateTokenAPIView(BaseAPIView):
    def post(self, request):
        data = {
            "email": settings.USER_EMAIL,
            "user_token": settings.USER_TOKEN,
        }
        response = requests.post(
            f"{settings.TERRA_AI_EXCHANGE_API_URL}/update_token/", json=data
        )
        if not response.json().get("new_token"):
            return BaseResponseErrorGeneral("Не удалось обновить токен пользователя")
        new_token = response.json().get("new_token")
        utils.update_env_file(token=new_token)
        return BaseResponseSuccess(data={"new_token": new_token})
