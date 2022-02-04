from django.conf import settings

from apps.api import decorators, remote
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.profile.serializers import SaveSerializer


class SaveAPIView(BaseAPIView):
    @decorators.serialize_data(SaveSerializer)
    def post(self, request, serializer, **kwargs):
        data = serializer.validated_data
        remote.request("/update/", data)
        settings.USER = {
            **(settings.USER or {}),
            "first_name": data.get("first_name"),
            "last_name": data.get("last_name"),
        }
        return BaseResponseSuccess()


class UpdateTokenAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        token = remote.request("/update/token/")
        settings.USER = {**(settings.USER or {}), "token": token}
        return BaseResponseSuccess({"new_token": token})
