from .serializers import CascadeGetSerializer
from apps.plugins.project import project_path
from terra_ai.agent import agent_exchange

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
)


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("cascades_info", path=project_path.cascades).native()
        )
