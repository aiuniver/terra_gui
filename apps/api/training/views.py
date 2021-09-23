from dict_recursive_update import recursive_update
from pydantic import ValidationError

from apps.media.utils import path_hash

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.exceptions.base import TerraBaseException
from terra_ai.data.training.train import TrainData

from apps.plugins.project import project_path

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorGeneral,
    BaseResponseErrorFields,
)


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            training_base = request.project.training.base.native()
            request_outputs = dict(
                map(
                    lambda item: (item.get("id"), item),
                    request.data.get("architecture", {})
                    .get("parameters", {})
                    .get("outputs", []),
                )
            )
            outputs = (
                request.project.training.base.architecture.parameters.outputs.native()
            )
            for index, item in enumerate(outputs):
                outputs[index] = recursive_update(
                    item, request_outputs.get(item.get("id"), {})
                )
            training_base = recursive_update(training_base, request.data)
            training_base["architecture"]["parameters"]["outputs"] = outputs
            request.project.training.base = TrainData(**training_base)
            data = {
                "dataset": request.project.dataset,
                "model": request.project.model,
                "training_path": project_path.training,
                "dataset_path": project_path.datasets,
                "params": request.project.training.base,
                "initial_config": request.project.training.interactive,
            }
            agent_exchange("training_start", **data)
            request.project.training.set_state()
            return BaseResponseSuccess(
                {"state": request.project.training.state.native()}
            )
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except (TerraBaseException, ExchangeBaseException) as error:
            return BaseResponseErrorGeneral(str(error))


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_stop")
            request.project.training.set_state()
            return BaseResponseSuccess(
                {"state": request.project.training.state.native()}
            )
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_clear")
            request.project.training.set_state()
            return BaseResponseSuccess(
                {"state": request.project.training.state.native()}
            )
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("training_interactive", **request.data)
            return BaseResponseSuccess(
                {"state": request.project.training.state.native()}
            )
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class ProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            data = agent_exchange("training_progress").native()
            try:
                for _, result in (
                    data.get("data", {})
                    .get("train_data", {})
                    .get("intermediate_result", {})
                    .items()
                ):
                    for _, layer in result.get("initial_data", {}).items():
                        for item in layer.get("data", []):
                            item.update({"value": path_hash(path=item.get("value"))})
            except Exception:
                pass
            request.project.training.set_state()
            data.update({"state": request.project.training.state.native()})
            return BaseResponseSuccess(data)
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_save")
        return BaseResponseSuccess()
