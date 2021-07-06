import os

from apps.plugins.terra import terra_exchange

from ..base import BaseAPIView, BaseResponse
from .data import DatasetsSourcesList


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        __items = DatasetsSourcesList()
        __sources_path = terra_exchange.project.gd.datasets_sources
        for filename in os.listdir(__sources_path):
            try:
                __items.append({"value": os.path.join(__sources_path, filename)})
            except Exception:
                pass
        return BaseResponse(data=__items.list())
