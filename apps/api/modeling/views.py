from pydantic import ValidationError

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import ModelLoadSerializer


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            request.project.model = agent_exchange(
                "model_load", **serializer.validated_data
            )
            request.project.save()
            return BaseResponseSuccess(request.project.model.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=str(data_path.modeling)).native()
        )
