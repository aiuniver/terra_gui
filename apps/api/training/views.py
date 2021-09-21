from pydantic import ValidationError

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.exceptions.base import TerraBaseException
from terra_ai.data.training.train import TrainData
from terra_ai.data.training.extra import ArchitectureChoice

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
            request.data["architecture"]["type"] = ArchitectureChoice.Basic
            request.project.training.base = TrainData(**request.data)
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
            request.project.training.set_state()
            data.update({"state": request.project.training.state.native()})
            return BaseResponseSuccess(data)
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_save")
        return BaseResponseSuccess()
