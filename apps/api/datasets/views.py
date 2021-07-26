from pydantic import ValidationError

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException

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
            request.project.dataset = agent_exchange(
                "dataset_choice",
                path=str(data_path.datasets),
                **serializer.validated_data
            )
            return BaseResponseSuccess(request.project.dataset.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class InfoAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_info", path=str(data_path.datasets)).native()
        )


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
        return BaseResponseSuccess(
            data=agent_exchange("dataset_source_load_progress").native()
        )


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_sources", path=str(data_path.sources)).native()
        )
