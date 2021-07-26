from pydantic import ValidationError

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.data.datasets.dataset import DatasetData

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import SourceLoadSerializer, ChoiceSerializer


class ChoiceAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ChoiceSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            data = agent_exchange(
                "dataset_choice",
                path=str(data_path.datasets),
                **serializer.validated_data
            )
            request.project.dataset = DatasetData(**data)
            return BaseResponseSuccess(data)
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class InfoAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange("datasets_info", path=str(data_path.datasets))
        return BaseResponseSuccess(data)


class SourceLoadAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        serializer = SourceLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            data = agent_exchange("dataset_source_load", **serializer.validated_data)
            return BaseResponseSuccess(data)
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class SourceLoadProgressAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(data=agent_exchange("dataset_source_load_progress"))


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange("datasets_sources", path=str(data_path.sources))
        return BaseResponseSuccess(data)
