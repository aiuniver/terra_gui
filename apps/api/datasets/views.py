from pathlib import Path

from terra_ai.settings import TERRA_PATH
from terra_ai.data.extra import FileManagerItem
from terra_ai.data.datasets.creation import CreationData, CreationValidateBlocksData

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.datasets.serializers import (
    VersionsSerializer,
    ChoiceSerializer,
    SourceLoadSerializer,
    SourceSegmentationClassesAutosearchSerializer,
    CreateSerializer,
    CreateValidateSerializer,
    DeleteSerializer,
)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        datasets = self.terra_exchange("datasets_info")
        return BaseResponseSuccess(
            {
                "datasets": datasets.native(),
                "groups": list(
                    map(lambda item: {"alias": item.alias, "name": item.name}, datasets)
                ),
                "tags": [
                    {
                        "name": "Tags",
                        "items": datasets.tags,
                    }
                ],
            }
        )


class VersionsAPIView(BaseAPIView):
    @decorators.serialize_data(VersionsSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "datasets_versions", **serializer.validated_data
            ).native()
        )


class ChoiceAPIView(BaseAPIView):
    @decorators.serialize_data(ChoiceSerializer)
    def post(self, request, serializer, **kwargs):
        self.terra_exchange("dataset_choice", **serializer.validated_data)
        return BaseResponseSuccess()


class ChoiceProgressAPIView(BaseAPIView):
    @decorators.progress_error("dataset_choice")
    def post(self, request, progress, **kwargs):
        if progress.finished and progress.data and progress.data.get("dataset"):
            request.project.set_dataset(**progress.data)
            progress.data = request.project.dataset.native()
        return BaseResponseSuccess(progress.native())


class SourcesAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "datasets_sources", path=str(TERRA_PATH.sources)
            ).native()
        )


class SourceLoadAPIView(BaseAPIView):
    @decorators.serialize_data(SourceLoadSerializer)
    def post(self, request, serializer, **kwargs):
        self.terra_exchange("dataset_source_load", **serializer.validated_data)
        return BaseResponseSuccess()


class SourceLoadProgressAPIView(BaseAPIView):
    @decorators.progress_error("dataset_source_load")
    def post(self, request, progress, **kwargs):
        if progress.finished and progress.data:
            source_path = Path(progress.data).absolute()
            file_manager = FileManagerItem(path=source_path).native().get("children")
            progress.data = {
                "file_manager": file_manager,
                "source_path": source_path,
            }
        return BaseResponseSuccess(progress.native())


class SourceSegmentationClassesAutoSearchAPIView(BaseAPIView):
    @decorators.serialize_data(SourceSegmentationClassesAutosearchSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_source_segmentation_classes_auto_search",
                path=request.data.get("path"),
                **serializer.validated_data,
            )
        )


class SourceSegmentationClassesAnnotationAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_source_segmentation_classes_annotation",
                path=request.data.get("path"),
            )
        )


class CreateAPIView(BaseAPIView):
    @decorators.serialize_data(CreateSerializer)
    def post(self, request, serializer, **kwargs):
        data = CreationData(**serializer.data)
        self.terra_exchange("dataset_create", creation_data=data)
        return BaseResponseSuccess()


class CreateProgressAPIView(BaseAPIView):
    @decorators.progress_error("create_dataset")
    def post(self, request, progress, **kwargs):
        return BaseResponseSuccess(progress.native())


class CreateValidateAPIView(BaseAPIView):
    @decorators.serialize_data(CreateValidateSerializer)
    def post(self, request, serializer, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "dataset_create_validate",
                data=CreationValidateBlocksData(**serializer.validated_data),
            )
        )


class DeleteAPIView(BaseAPIView):
    @decorators.serialize_data(DeleteSerializer)
    def post(self, request, serializer, **kwargs):
        data = serializer.validated_data
        self.terra_exchange("dataset_delete", path=str(TERRA_PATH.datasets), **data)
        if request.project.dataset and (
            request.project.dataset.alias == data.get("alias")
        ):
            request.project.clear_dataset()
        return BaseResponseSuccess()
