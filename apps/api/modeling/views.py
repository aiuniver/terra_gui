from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess


class ModelAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange("model", **request.data)
        return BaseResponseSuccess(data)


class ModelsAPIView(BaseAPIView):
    def get(self, request, **kwargs):
        data = agent_exchange(
            "models", path=str(request.project.path.modeling.absolute())
        )
        return BaseResponseSuccess(data)
