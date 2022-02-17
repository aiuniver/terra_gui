from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class AutosearchAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.AutosearchSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_source_segmentation_classes_auto_search",
                path=request.data.get("path"),
                **serializer.validated_data,
            )
        )


class AnnotationAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_source_segmentation_classes_annotation",
                path=request.data.get("path"),
            )
        )
