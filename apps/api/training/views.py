from dict_recursive_update import recursive_update

from terra_ai.agent import agent_exchange
from terra_ai.data.training.train import TrainData, InteractiveData
from terra_ai.data.training.extra import StateStatusChoice

from apps.plugins.project import project_path
from apps.plugins.frontend import defaults_data

from ..base import BaseAPIView, BaseResponseSuccess


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        training_base = request.project.training.base.native()
        request_outputs = dict(
            map(
                lambda item: (item.get("id"), item),
                request.data.get("architecture", {})
                .get("parameters", {})
                .get("outputs", []),
            )
        )
        outputs = request.project.training.base.architecture.parameters.outputs.native()
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
        return BaseResponseSuccess(
            {
                "interactive": request.project.training.interactive.native(),
                "state": request.project.training.state.native(),
            }
        )


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_stop")
        request.project.training.set_state()
        return BaseResponseSuccess({"state": request.project.training.state.native()})


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_clear")
        request.project.training.set_state()
        request.project.training.result = None
        return BaseResponseSuccess({"state": request.project.training.state.native()})


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        config = InteractiveData(**request.data)
        request.project.training.interactive = config
        training_data: dict = None
        if request.project.training.state.status != StateStatusChoice.no_train:
            training_data = agent_exchange("training_interactive", config=config)
            request.project.training.result = (
                training_data.get("train_data") if training_data else None
            )
        return BaseResponseSuccess(training_data)


class ProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        data = agent_exchange("training_progress").native()
        request.project.training.set_state()
        data.update({"state": request.project.training.state.native()})
        _finished = data.get("finished")
        if _finished:
            request.project.deploy = agent_exchange("deploy_presets")
        request.project.training.result = data.get("data", {}).get("train_data", {})
        if _finished:
            request.project.save()
        return BaseResponseSuccess(data)


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("training_save")
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.update_training_base(request.data)
        return BaseResponseSuccess(defaults_data.training.native())
