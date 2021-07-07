from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponse


class InfoAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange(
            "get_datasets_info",
            path=str(request.project.path.datasets.absolute()),
        )
        return BaseResponse(data=data)


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange(
            "get_datasets_sources",
            path=str(request.project.path.sources.absolute()),
        )
        return BaseResponse(data=data)
