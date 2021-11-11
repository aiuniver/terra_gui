from apps.api import cascades
from .serializers import CascadeGetSerializer, UpdateSerializer
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


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = request.project.cascade
        data = serializer.validated_data
        cascade_data = cascade.native()
        cascade_data.update(data)
        cascade = agent_exchange("cascade_update", cascade=cascade_data)
        request.project.set_cascade(cascade)
        return BaseResponseSuccess()


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_cascade()
        return BaseResponseSuccess(request.project.cascade.native())
