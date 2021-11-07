from terra_ai.agent import agent_exchange
from terra_ai.data.training.train import InteractiveData
from terra_ai.data.training.extra import StateStatusChoice

from apps.plugins.frontend import defaults_data

from apps.api.base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields

from . import serializers


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        if (
            request.project.training.state.status == StateStatusChoice.stopped
            or request.project.training.state.status == StateStatusChoice.trained
        ):
            request.project.training.state.set(StateStatusChoice.addtrain)
        else:
            request.project.training.state.set(StateStatusChoice.training)
        request.project.set_training_base(request.data)
        agent_exchange(
            "training_start",
            **{
                "dataset": request.project.dataset,
                "model": request.project.model,
                "training": request.project.training,
            }
        )
        return BaseResponseSuccess(
            {
                "form": defaults_data.training.native(),
                "interactive": request.project.training.interactive.native(),
                "state": request.project.training.state.native(),
            }
        )


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        training_base = request.project.training.base.native()
        agent_exchange("training_stop", training=request.project.training)
        request.project.set_training_base(training_base)
        return BaseResponseSuccess(
            {
                "form": defaults_data.training.native(),
                "state": request.project.training.state.native(),
            }
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        name = request.project.training.name
        agent_exchange("training_clear", training=request.project.training)
        request.project.clear_training(name)
        return BaseResponseSuccess(
            {
                "form": defaults_data.training.native(),
                "state": request.project.training.state.native(),
            }
        )


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
        current_state = request.project.training.state.status
        data = agent_exchange("training_progress").native()
        data.update({"state": request.project.training.state.native()})
        if current_state != request.project.training.state.status:
            request.project.set_training_base(request.project.training.base.native())
            data.update({"form": defaults_data.training.native()})
        _finished = data.get("finished")
        if _finished:
            request.project.save()
        return BaseResponseSuccess(data)


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.training.save(**serializer.validated_data)
        agent_exchange("training_save")
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.set_training_base(request.data)
        return BaseResponseSuccess(
            {
                "form": defaults_data.training.native(),
                "data": request.project.training.native(),
            }
        )
