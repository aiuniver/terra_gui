from pydantic import ValidationError

from apps.plugins.project import data_path
from apps.plugins.project import exceptions as project_exceptions
from terra_ai.exceptions.base import TerraBaseException
from terra_ai.agent import agent_exchange

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import (
    SourceLoadSerializer,
    ChoiceSerializer,
    CreateSerializer,
    DeleteSerializer,
    SourceSegmentationClassesAutosearchSerializer,
)


class ChoiceAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ChoiceSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            agent_exchange(
                "dataset_choice",
                path=str(data_path.datasets),
                **serializer.validated_data,
            )
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class ChoiceProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("dataset_choice_progress")
        if progress.finished and progress.data:
            try:
                request.project.set_dataset(progress.data)
            except project_exceptions.ProjectException as error:
                return BaseResponseErrorGeneral(str(error))
        return BaseResponseSuccess(data=progress.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_info", path=str(data_path.datasets)).native()
        )


class SourceLoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = SourceLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            agent_exchange("dataset_source_load", **serializer.validated_data)
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class SourceLoadProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            data=agent_exchange("dataset_source_load_progress").native()
        )


class SourceSegmentationClassesAutosearchAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = SourceSegmentationClassesAutosearchSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            return BaseResponseSuccess(
                agent_exchange(
                    "dataset_source_segmentation_classes_autosearch",
                    path=request.data.get("path"),
                    **serializer.validated_data,
                )
            )
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class SourceSegmentationClassesAnnotationAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            return BaseResponseSuccess(
                agent_exchange(
                    "dataset_source_segmentation_classes_annotation",
                    path=request.data.get("path"),
                )
            )
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            return BaseResponseSuccess(
                data=agent_exchange("dataset_create", **serializer.data).native()
            )
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class SourcesAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("datasets_sources", path=str(data_path.sources)).native()
        )


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = DeleteSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            agent_exchange(
                "dataset_delete",
                path=str(data_path.datasets),
                **serializer.validated_data,
            )
            if request.project.dataset and (
                request.project.dataset.alias == serializer.validated_data.get("alias")
            ):
                request.project.dataset = None
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except TerraBaseException as error:
            return BaseResponseErrorGeneral(str(error))
