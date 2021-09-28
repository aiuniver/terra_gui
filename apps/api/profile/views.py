import requests

from django.conf import settings

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import SaveSerializer


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
            return BaseResponseSuccess()
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class UpdateTokenAPIView(BaseAPIView):
    def post(self):
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
            return BaseResponseSuccess(data={"new_token": new_token})
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
