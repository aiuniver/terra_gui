import json

from apps.plugins.frontend import presets
from apps.plugins.frontend.defaults import DefaultsData

from ..base import BaseAPIView, BaseResponseSuccess


class ConfigAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(json.loads(request.project.json()))


class DefaultsAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = DefaultsData(**presets.defaults.Defaults)
        return BaseResponseSuccess(json.loads(data.json()))
