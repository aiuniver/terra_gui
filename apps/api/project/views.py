from apps.plugins.project import project_path, data_path
from terra_ai.agent import agent_exchange
from terra_ai.data.projects.project import ProjectPathData
from .serializers import (
    NameSerializer,
    SaveSerializer,
    LoadSerializer,
    DeleteSerializer,
)
from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
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
        agent_exchange(
            "project_save",
            source=project_path.base,
            target=data_path.projects,
            name=serializer.validated_data.get("name"),
            overwrite=serializer.validated_data.get("overwrite"),
        )
        return BaseResponseSuccess()


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
        agent_exchange(
            "project_load",
            source=serializer.validated_data.get("value"),
            target=project_path.base,
        )
        request.project.load()
        return BaseResponseSuccess()


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = DeleteSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        project = ProjectPathData(path=serializer.validated_data.get("path"))
        project.path.unlink()
        return BaseResponseSuccess()
