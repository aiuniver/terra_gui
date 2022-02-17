from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class ChoiceAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.ChoiceSerializer)
    def post(self, request, serializer, **kwargs):
        self.terra_exchange("dataset_choice", **serializer.validated_data)
        return BaseResponseSuccess()


class ProgressAPIView(BaseAPIView):
    @decorators.progress_error("dataset_choice")
    def post(self, request, progress, **kwargs):
        if progress.finished and progress.data and progress.data.get("dataset"):
            request.project.set_dataset(**progress.data)
            progress.data = request.project.dataset.native()
        return BaseResponseSuccess(progress.native())
