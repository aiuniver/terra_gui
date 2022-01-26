from pydantic import BaseModel

from terra_ai.data.training.extra import StateStatusChoice

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.training.serializers import SaveSerializer
from apps.plugins.frontend import defaults_data


class TrainingResponseData(BaseModel):
    base: dict
    form: dict
    state: dict
    interactive: dict
    result: dict
    progress: dict

    def __init__(self, project, defaults, **kwargs):
        kwargs.update(
            {
                "base": project.training.base.native(),
                "form": defaults.training.native(),
                "state": project.training.state.native(),
                "interactive": project.training.interactive.native(),
                "result": project.training.result,
                "progress": project.training.progress,
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
        self.terra_exchange(
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
        self.terra_exchange("training_stop", training=request.project.training)
        request.project.set_training_base(training_base)
        request.project.training.save(request.project.training.name)
        request.project.save_config()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        name = request.project.training.name
        self.terra_exchange("training_clear", training=request.project.training)
        request.project.clear_training(name)
        request.project.training.save(request.project.training.name)
        request.project.save_config()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class InteractiveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.training.set_interactive(data=request.data)
        self.terra_exchange("training_interactive", training=request.project.training)
        request.project.training.save(request.project.training.name)
        request.project.save_config()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class ProgressAPIView(BaseAPIView):
    @decorators.progress_error("training")
    def post(self, request, progress, **kwargs):
        if progress.finished and progress.percent == 100:
            progress.percent = 0
        request.project.training.progress = progress.native()
        if progress.finished:
            request.project.set_training_base(request.project.training.base.native())
            request.project.training.save(request.project.training.name)
            defaults_data.update_models(request.project.trainings)
            request.project.save_config()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )


class SaveAPIView(BaseAPIView):
    @decorators.serialize_data(SaveSerializer)
    def post(self, request, serializer, **kwargs):
        request.project.training.save(**serializer.validated_data)
        defaults_data.update_models(request.project.trainings)
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.set_training_base(request.data)
        request.project.training.save(request.project.training.name)
        request.project.save_config()
        return BaseResponseSuccess(
            TrainingResponseData(request.project, defaults_data).dict()
        )
