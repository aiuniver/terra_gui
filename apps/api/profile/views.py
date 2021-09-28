import requests

from django.conf import settings

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import SaveSerializer
from . import utils


class SaveAPIView(BaseAPIView):
    def post(self, request):
        serializer = SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
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
                return BaseResponseErrorGeneral(
                    "Не удалось обновить данные пользователя"
                )
            utils.update_env_file(**serializer.validated_data)
            return BaseResponseSuccess()
        except Exception as error:
            return BaseResponseErrorGeneral(str(error))


class UpdateTokenAPIView(BaseAPIView):
    def post(self, request):
        try:
            data = {
                "email": settings.USER_EMAIL,
                "user_token": settings.USER_TOKEN,
            }
            response = requests.post(
                f"{settings.TERRA_AI_EXCHANGE_API_URL}/update_token/", json=data
            )
            if not response.json().get("new_token"):
                return BaseResponseErrorGeneral(
                    "Не удалось обновить токен пользователя"
                )
            new_token = response.json().get("new_token")
            utils.update_env_file(token=new_token)
            return BaseResponseSuccess(data={"new_token": new_token})
        except Exception as error:
            return BaseResponseErrorGeneral(str(error))
