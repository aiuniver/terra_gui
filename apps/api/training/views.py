from pydantic import BaseModel

from terra_ai.agent import agent_exchange
from terra_ai.data.training.extra import StateStatusChoice

from apps.plugins.frontend import defaults_data

from apps.api.base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields

from . import serializers


class TrainingResponseData(BaseModel):
    form: dict
    state: dict
    interactive: dict
    result: dict
    progress: dict

    def __init__(self, project, defaults, **kwargs):
        kwargs.update(
            {
                "form": defaults.training.native(),
                "state": project.training.state.native(),
                "interactive": project.training.interactive.native(),
                "result": project.training.result,
                "progress": project.training.progress.native(),
            }
        )
        super().__init__(**kwargs)


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
            TrainingResponseData(request.project, defaults_data).dict()
        )


class StopAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        training_base = request.project.training.base.native()
        agent_exchange("training_stop", training=request.project.training)
        request.project.set_training_base(training_base)
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        name = request.project.training.name
        agent_exchange("training_clear", training=request.project.training)
        request.project.clear_training(name)
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.training.set_interactive(request.data)
        agent_exchange("training_interactive", request.project.training)
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class ProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        # current_state = request.project.training.state.status
        progress = agent_exchange("training_progress")
        # progress.update({"state": request.project.training.state.native()})
        # if current_state == request.project.training.state.status:
        #     request.project.set_training_base(request.project.training.base.native())
        #     progress.update({"form": defaults_data.training.native()})
        if progress.finished:
            request.project.training.save()
            request.project.save()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.training.save(**serializer.validated_data)
        # agent_exchange("training_save")
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.set_training_base(request.data)
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )
