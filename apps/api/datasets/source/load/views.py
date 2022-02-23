from pathlib import Path

from terra_ai.data.extra import FileManagerItem
from terra_ai.data.presets.datasets import DatasetCreationArchitecture
from terra_ai.data.datasets.creation import (
    CreationData,
    DatasetCreationArchitectureData,
)

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
            extra = {**progress.data}
            progress.data = (
                FileManagerItem(path=progress.data.get("source").get("path"))
                .native()
                .get("children")
            )
            extra.update(
                {
                    "version": DatasetCreationArchitectureData(
                        **DatasetCreationArchitecture.get(extra.get("architecture"))
                    ).native()
                }
            )
            request.project.set_dataset_creation(CreationData(stage=2, **extra))
        return BaseResponseSuccess(progress.native())
