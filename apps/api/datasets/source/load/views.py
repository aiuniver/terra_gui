from pathlib import Path

from terra_ai.data.extra import FileManagerItem
from terra_ai.data.presets.datasets import DatasetCreationArchitecture
from terra_ai.data.datasets.creation import DatasetCreationArchitectureData

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess

from . import serializers


class LoadAPIView(BaseAPIView):
    @decorators.serialize_data(serializers.LoadSerializer)
    def post(self, request, serializer, **kwargs):
        self.terra_exchange("dataset_source_load", **serializer.validated_data)
        return BaseResponseSuccess()


class ProgressAPIView(BaseAPIView):
    @decorators.progress_error("dataset_source_load")
    def post(self, request, progress, **kwargs):
        if progress.finished and progress.data:
            source_path = Path(progress.data.get("path")).absolute()
            file_manager = FileManagerItem(path=source_path).native().get("children")
            progress.data = {
                "file_manager": file_manager,
                "source_path": source_path,
                "blocks": DatasetCreationArchitectureData(
                    **DatasetCreationArchitecture.get(progress.data.get("architecture"))
                ).native(),
            }
        return BaseResponseSuccess(progress.native())
