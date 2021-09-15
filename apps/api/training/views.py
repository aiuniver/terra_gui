from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.data.training.train import TrainData

from apps.plugins.project import project_path
from terra_ai.exceptions.base import TerraBaseException

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorGeneral


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            data = {
                "dataset": request.project.dataset,
                "model": request.project.model,
                "training_path": project_path.training,
                "dataset_path": project_path.datasets,
                "params": TrainData(**request.data),
            }
            return BaseResponseSuccess(agent_exchange("training_start", **data))
        except (TerraBaseException, ExchangeBaseException) as error:
            return BaseResponseErrorGeneral(str(error))


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_stop")
            return BaseResponseSuccess()
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_clear")
            return BaseResponseSuccess()
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_interactive", **request.data)
            return BaseResponseSuccess()
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class ProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            return BaseResponseSuccess(agent_exchange("training_progress").native())
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))