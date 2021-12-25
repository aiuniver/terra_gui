import requests

from django.conf import settings

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.profile.utils import update_env_file
from apps.api.profile.serializers import SaveSerializer


class SaveAPIView(BaseAPIView):
    @decorators.serialize_data(SaveSerializer)
    def post(self, request, serializer, **kwargs):
        data = dict(serializer.validated_data)
        data.update(
            {
                "email": settings.USER_EMAIL,
                "user_token": settings.USER_TOKEN,
            }
        )

        response = requests.post(f"{settings.TERRA_API_URL}/update/", json=data)
        response_data = response.json()
        if not response_data.get("success"):
            raise ValueError("Не удалось обновить данные пользователя")

        update_env_file(**serializer.validated_data)
        return BaseResponseSuccess()


class UpdateTokenAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        data = {
            "email": settings.USER_EMAIL,
            "user_token": settings.USER_TOKEN,
        }

        response = requests.post(f"{settings.TERRA_API_URL}/update_token/", json=data)
        response_data = response.json()
        if not response_data.get("new_token"):
            raise ValueError("Не удалось обновить токен пользователя")

        new_token = response_data.get("new_token")
        update_env_file(token=new_token)
        return BaseResponseSuccess({"new_token": new_token})
