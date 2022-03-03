from terra_ai.data.datasets.extra import LayerGroupChoice
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
        if progress.finished and int(progress.percent) == 100:
            request.project.set_dataset_creation(CreationData())
        return BaseResponseSuccess(progress.native())


class ValidateAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.ValidateSerializer)
    def post(self, request, serializer, **kwargs):
        data = CreationValidateBlocksData(**serializer.validated_data)
        errors = self.terra_exchange("dataset_create_validate", data=data)
        if not list(filter(None, errors.values())):
            creation_data = request.project.dataset_creation.native()
            if data.type == LayerGroupChoice.inputs:
                creation_data.update({"stage": 3})
            elif data.type == LayerGroupChoice.outputs:
                creation_data.update({"stage": 4})
            if not creation_data.get("version"):
                creation_data.update({"version": {}})
            creation_data.get("version").update({data.type: data.items.native()})
            request.project.set_dataset_creation(CreationData(**creation_data))
        return BaseResponseSuccess(errors)
