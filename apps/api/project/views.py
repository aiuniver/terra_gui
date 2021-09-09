from terra_ai.agent import agent_exchange

from apps.plugins.project import project_path, data_path

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import NameSerializer


class NameAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = NameSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.name = serializer.validated_data.get("name")
        return BaseResponseSuccess()


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.reset()
        request.project.save()
        return BaseResponseSuccess()


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.save()
        # agent_exchange("project_save", **serializer.validated_data)
        return BaseResponseSuccess()


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess()
