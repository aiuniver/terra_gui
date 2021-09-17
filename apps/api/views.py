import json

from django.conf import settings

from apps.plugins.frontend import defaults_data

from .base import BaseAPIView, BaseResponseSuccess


class NotFoundAPIView(BaseAPIView):
    pass


class ConfigAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            {
                "defaults": json.loads(defaults_data.json()),
                "project": json.loads(request.project.json()),
                "user": {
                    "login": settings.USER_LOGIN,
                    "first_name": settings.USER_NAME,
                    "last_name": settings.USER_LASTNAME,
                },
            }
        )
