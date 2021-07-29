from pydantic import ValidationError

from apps.plugins.project import data_path, project_path
from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import ModelLoadSerializer


class ModelLoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            data = agent_exchange(
                "model_load",
                destination=project_path.modeling,
                **serializer.validated_data
            )
            return BaseResponseSuccess(data)
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class ModelLoadProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("model_load_progress").native()
        if progress.finished:
            request.project.model = agent_exchange(
                "dataset_choice",
                path=str(data_path.datasets),
                **serializer.validated_data
            )
            request.project.save()
        return BaseResponseSuccess(progress)


class ModelsAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=str(data_path.modeling)).native()
        )
