from pydantic import ValidationError

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import ModelLoadSerializer


class ModelLoadAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        serializer = ModelLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            data = agent_exchange("model_load", **serializer.validated_data)
            return BaseResponseSuccess(data)
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class ModelLoadProgressAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(agent_exchange("model_load_progress").native())


class ModelsAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=str(data_path.modeling)).native()
        )
