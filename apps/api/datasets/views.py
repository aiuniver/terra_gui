from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponse


class SourcesAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        sources = agent_exchange(
            "get_datasets_sources",
            pathdir=str(request.project.path.sources.absolute()),
        )
        return BaseResponse(data=sources)
