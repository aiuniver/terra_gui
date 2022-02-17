from terra_ai.settings import TERRA_PATH

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        datasets = self.terra_exchange("datasets_info")
        return BaseResponseSuccess(
            {
                "datasets": datasets.native(),
                "groups": dict(map(lambda item: (item.alias, item.name), datasets)),
                "tags": datasets.tags.native(),
            }
        )


class VersionsAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.VersionsSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "datasets_versions", **serializer.validated_data
            ).native()
        )


class DeleteAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.DeleteSerializer)
    def post(self, request, serializer, **kwargs):
        data = serializer.validated_data
        self.terra_exchange("dataset_delete", path=str(TERRA_PATH.datasets), **data)
        if request.project.dataset and (
            request.project.dataset.alias == data.get("alias")
        ):
            request.project.clear_dataset()
        return BaseResponseSuccess()
