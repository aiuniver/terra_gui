import json

from apps.plugins.frontend import presets
from apps.plugins.frontend.defaults import DefaultsData

from .base import BaseAPIView, BaseResponseSuccess


class NotFoundAPIView(BaseAPIView):
    pass


class ConfigAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            {
                "defaults": json.loads(
                    DefaultsData(**presets.defaults.Defaults).json()
                ),
                "project": json.loads(request.project.json()),
            }
        )
