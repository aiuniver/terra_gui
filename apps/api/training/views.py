from pydantic import ValidationError

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.data.training.train import TrainData, InteractiveData

from apps.plugins.project import project_path
from terra_ai.exceptions.base import TerraBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorGeneral,
    BaseResponseErrorFields,
)


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        architecture = request.data.get("architecture", {})
        architecture.update({"type": "Basic"})
        request.data.update({"architecture": architecture})
        for layer_data in architecture.get("parameters", {}).get("outputs", []):
            if not layer_data:
                continue
            layer = request.project.model.outputs.get(layer_data.get("id"))
            if not layer:
                continue
            layer_data.update({"task": layer.task.value})
        try:
            data = {
                "dataset": request.project.dataset,
                "model": request.project.model,
                "training_path": project_path.training,
                "dataset_path": project_path.datasets,
                "params": TrainData(**request.data),
                "initial_config": InteractiveData(
                    **{
                        "loss_graphs": [
                            {
                                "id": 1,
                            }
                        ],
                    }
                ),
            }
            return BaseResponseSuccess(agent_exchange("training_start", **data))
        except ValidationError as error:
            return BaseResponseErrorFields(error)
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
