import shutil

from terra_ai.settings import TERRA_PATH
from terra_ai.agent import agent_exchange
from terra_ai.data.datasets.creation import CreationData

from apps.api.base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from . import serializers


class ChoiceAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        serializer = serializers.ChoiceSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        agent_exchange(
            "dataset_choice",
            custom_path=TERRA_PATH.datasets,
            **serializer.validated_data,
        )
        return BaseResponseSuccess()


class ChoiceProgressAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        progress = agent_exchange("dataset_choice_progress")
        if progress.finished and progress.data and progress.data.get("info"):
            request.project.set_dataset(**progress.data)
            progress.data = request.project.dataset.native()
        if progress.success:
            return BaseResponseSuccess(data=progress.native())
        else:
            return BaseResponseErrorGeneral(progress.error, data=progress.native())


class InfoAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_info", path=TERRA_PATH.datasets).native()
        )


class SourceLoadAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        serializer = serializers.SourceLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        agent_exchange("dataset_source_load", **serializer.validated_data)
        return BaseResponseSuccess()


class SourceLoadProgressAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        progress = agent_exchange("dataset_source_load_progress")
        if progress.success:
            return BaseResponseSuccess(data=progress.native())
        else:
            return BaseResponseErrorGeneral(progress.error, data=progress.native())


class SourceSegmentationClassesAutoSearchAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        serializer = serializers.SourceSegmentationClassesAutosearchSerializer(
            data=request.data
        )
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        return BaseResponseSuccess(
            agent_exchange(
                "dataset_source_segmentation_classes_auto_search",
                path=request.data.get("path"),
                **serializer.validated_data,
            )
        )


class SourceSegmentationClassesAnnotationAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange(
                "dataset_source_segmentation_classes_annotation",
                path=request.data.get("path"),
            )
        )


class CreateAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        serializer = serializers.CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        data = CreationData(**serializer.data)
        agent_exchange("dataset_create", creation_data=data)
        return BaseResponseSuccess()


class CreateProgressAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        progress = agent_exchange("dataset_create_progress")
        if progress.success:
            return BaseResponseSuccess(progress.native())
        else:
            shutil.rmtree(progress.data.get("path"), ignore_errors=True)
            return BaseResponseErrorGeneral(progress.error, data=progress.native())


class SourcesAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_sources", path=str(TERRA_PATH.sources)).native()
        )


class DeleteAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        serializer = serializers.DeleteSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        agent_exchange(
            "dataset_delete",
            path=str(TERRA_PATH.datasets),
            **serializer.validated_data,
        )
        if request.project.dataset and (
            request.project.dataset.alias == serializer.validated_data.get("alias")
        ):
            request.project.clear_dataset()
        return BaseResponseSuccess()
