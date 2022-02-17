from terra_ai.data.datasets.creation import CreationData, CreationValidateBlocksData

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class CreateAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.CreateSerializer)
    def post(self, request, serializer, **kwargs):
        data = CreationData(**serializer.data)
        self.terra_exchange("dataset_create", creation_data=data)
        return BaseResponseSuccess()


class ProgressAPIView(BaseAPIView):
    @decorators.progress_error("create_dataset")
    def post(self, request, progress, **kwargs):
        return BaseResponseSuccess(progress.native())


class ValidateAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.ValidateSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_create_validate",
                data=CreationValidateBlocksData(**serializer.validated_data),
            )
        )
