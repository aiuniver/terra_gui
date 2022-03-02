from terra_ai.data.datasets.creation import CreationData

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class VersionAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.VersionSerializer)
    def post(self, request, serializer, **kwargs):
        self.terra_exchange("dataset_create_version", **serializer.validated_data)
        return BaseResponseSuccess()


class ProgressAPIView(BaseAPIView):
    @decorators.progress_error("create_version")
    def post(self, request, progress, **kwargs):
        if progress.finished and int(progress.percent) == 100:
            request.project.set_dataset_creation(CreationData(**progress.data))
        return BaseResponseSuccess(progress.native())
