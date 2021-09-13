from terra_ai.agent import agent_exchange
from terra_ai.data.training.train import TrainData

from apps.plugins.project import project_path

from ..base import BaseAPIView, BaseResponseSuccess


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        data = {
            "dataset": request.project.dataset,
            "model": request.project.model,
            "training_path": project_path.training,
            "dataset_path": project_path.datasets,
            "params": TrainData(**request.data),
        }
        return BaseResponseSuccess(agent_exchange("training_start", **data))


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_stop")
        return BaseResponseSuccess()


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_clear")
        return BaseResponseSuccess()


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_interactive", **request.data)
        return BaseResponseSuccess()


class ProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(agent_exchange("training_progress").native())
