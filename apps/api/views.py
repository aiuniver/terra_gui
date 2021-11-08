import json

from django.conf import settings

from apps.plugins.frontend import defaults_data

from . import base


class NotFoundAPIView(base.BaseAPIView):
    pass


class ConfigAPIView(base.BaseAPIView):
    def post(self, request, **kwargs):
        return base.BaseResponseSuccess(
            {
                "defaults": json.loads(defaults_data.json()),
                "project": json.loads(request.project.frontend()),
                "user": {
                    "login": settings.USER_LOGIN,
                    "first_name": settings.USER_NAME,
                    "last_name": settings.USER_LASTNAME,
                    "email": settings.USER_EMAIL,
                    "token": settings.USER_TOKEN,
                },
            }
        )
