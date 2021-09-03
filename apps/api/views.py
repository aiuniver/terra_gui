import json

from django.conf import settings

from apps.plugins.frontend import presets
from apps.plugins.frontend.defaults import DefaultsData

from .base import BaseAPIView, BaseResponseSuccess


class NotFoundAPIView(BaseAPIView):
    pass


class ConfigAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        defaults = DefaultsData(**presets.defaults.Defaults)
        # for layer in request.project.model.outputs:
        #     print(layer)
        # print(defaults.training.base.outputs.json(indent=2))
        return BaseResponseSuccess(
            {
                "defaults": json.loads(defaults.json()),
                "project": json.loads(request.project.json()),
                "user": {
                    "login": settings.USER_LOGIN,
                    "first_name": settings.USER_NAME,
                    "last_name": settings.USER_LASTNAME,
                },
            }
        )
