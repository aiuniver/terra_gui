from pydantic import ValidationError

from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import SourceLoadSerializer


class InfoAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange(
            "get_datasets_info",
            path=str(request.project.path.datasets.absolute()),
        )
        return BaseResponseSuccess(data)


class SourceLoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = SourceLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            data = agent_exchange("get_dataset_source", **serializer.validated_data)
            return BaseResponseSuccess(data)
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange(
            "get_datasets_sources",
            path=str(request.project.path.sources.absolute()),
        )
        return BaseResponseSuccess(data)
