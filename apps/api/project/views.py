from pydantic import ValidationError

from terra_ai.agent import agent_exchange
from terra_ai.agent import exceptions as agent_exceptions
from terra_ai.data.projects.project import ProjectPathData

from apps.plugins.project import project_path, data_path

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import (
    NameSerializer,
    SaveSerializer,
    LoadSerializer,
    DeleteSerializer,
)


class NameAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = NameSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.name = serializer.validated_data.get("name")
        return BaseResponseSuccess(save_project=True)


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.reset()
        return BaseResponseSuccess()


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.name = serializer.validated_data.get("name")
        request.project.save()
        try:
            agent_exchange(
                "project_save",
                source=project_path.base,
                target=data_path.projects,
                name=serializer.validated_data.get("name"),
                overwrite=serializer.validated_data.get("overwrite"),
            )
            return BaseResponseSuccess()
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("projects_info", path=data_path.projects).native()
        )


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = LoadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            agent_exchange(
                "project_load",
                source=serializer.validated_data.get("value"),
                target=project_path.base,
            )
            request.project.load()
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except Exception as error:
            return BaseResponseErrorFields(str(error))


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = DeleteSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            project = ProjectPathData(path=serializer.validated_data.get("path"))
            project.path.unlink()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        return BaseResponseSuccess()
