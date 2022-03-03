from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from apps.plugins.frontend.choices import ArchitectureChoice

from . import serializers


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        datasets = self.terra_exchange("datasets_info")
        tags = datasets.tags.native()
        for item in tags:
            tag_name = item.get("name")
            item.update({"name": ArchitectureChoice[tag_name], "alias": tag_name})
        return BaseResponseSuccess(
            {
                "datasets": datasets.native(),
                "groups": dict(map(lambda item: (item.alias, item.name), datasets)),
                "tags": tags,
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
