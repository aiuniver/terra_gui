from pydantic import ValidationError

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.data.training.train import TrainData, InteractiveData
from terra_ai.data.training.extra import (
    LossGraphShowChoice,
    MetricGraphShowChoice,
    MetricChoice,
)

from apps.plugins.project import project_path, Project
from terra_ai.exceptions.base import TerraBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorGeneral,
    BaseResponseErrorFields,
)


class StartAPIView(BaseAPIView):
    def _get_training_defaults(self, data: dict, project: Project) -> TrainData:
        architecture = data.get("architecture", {})
        architecture.update({"type": "Basic"})
        data.update({"architecture": architecture})
        for layer_data in architecture.get("parameters", {}).get("outputs", []):
            if not layer_data:
                continue
            layer = project.model.outputs.get(layer_data.get("id"))
            if not layer:
                continue
            layer_data.update({"task": layer.task.value})
        return TrainData(**data)

    def _get_interactive_defaults(
        self, data: dict, project: Project
    ) -> InteractiveData:
        architecture = data.get("architecture", {})
        loss_graphs = []
        metric_graphs = []
        index = 0
        for layer_data in architecture.get("parameters", {}).get("outputs", []):
            if not layer_data:
                continue
            layer = project.model.outputs.get(layer_data.get("id"))
            print(layer_data)
            if not layer:
                continue
            metrics_list = layer_data.get("metrics", [])
            metric = metrics_list[0] if metrics_list else None
            index += 1
            loss_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.model,
                }
            )
            metric_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": MetricGraphShowChoice.model,
                    "show_metric": metric,
                }
            )
            index += 1
            loss_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.classes,
                }
            )
            metric_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": MetricGraphShowChoice.classes,
                    "show_metric": metric,
                }
            )
        return InteractiveData(
            **{
                "loss_graphs": loss_graphs,
                "metric_graphs": metric_graphs,
            }
        )

    def post(self, request, **kwargs):
        try:
            data = {
                "dataset": request.project.dataset,
                "model": request.project.model,
                "training_path": project_path.training,
                "dataset_path": project_path.datasets,
                "params": self._get_training_defaults(request.data, request.project),
                "initial_config": self._get_interactive_defaults(
                    request.data, request.project
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
