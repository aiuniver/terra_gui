from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class DeleteAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.DeleteSerializer)
    def post(self, request, serializer, **kwargs):
        data = serializer.validated_data
        self.terra_exchange("dataset_delete", **data)
        if request.project.dataset and (
            request.project.dataset.alias == data.get("alias")
        ):
            request.project.clear_dataset()
        return BaseResponseSuccess()


class DeleteVersionAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.DeleteVersionSerializer)
    def post(self, request, serializer, **kwargs):
        data = serializer.validated_data
        self.terra_exchange("dataset_delete_version", **data)
        if (
            request.project.dataset
            and request.project.dataset.alias == data.get("alias")
            and request.project.dataset.version.alias == data.get("version")
        ):
            request.project.clear_dataset()
        return BaseResponseSuccess()
