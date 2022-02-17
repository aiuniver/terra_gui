from terra_ai.settings import TERRA_PATH

from apps.api.base import BaseAPIView, BaseResponseSuccess


class SourceAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange("datasets_sources", path=str(TERRA_PATH.sources))
        )
