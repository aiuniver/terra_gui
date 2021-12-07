from terra_ai.settings import TERRA_PATH, PROJECT_PATH
from terra_ai.agent import agent_exchange
from terra_ai.project.loading import PROJECT_LOAD_NAME
from terra_ai.data.projects.project import ProjectPathData

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.project.serializers import (
    NameSerializer,
    SaveSerializer,
    LoadSerializer,
    DeleteSerializer,
)


class NameAPIView(BaseAPIView):
    @decorators.serialize_data(NameSerializer)
    def post(self, request, serializer, **kwargs):
        request.project.set_name(serializer.validated_data.get("name"))
        return BaseResponseSuccess()


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.create()
        return BaseResponseSuccess()


class SaveAPIView(BaseAPIView):
    @decorators.serialize_data(SaveSerializer)
    def post(self, request, serializer, **kwargs):
        request.project.set_name(serializer.validated_data.get("name"))
        request.project.save(serializer.validated_data.get("overwrite"))
        return BaseResponseSuccess()


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("projects_info", path=TERRA_PATH.projects).native()
        )


class LoadAPIView(BaseAPIView):
    @decorators.serialize_data(LoadSerializer)
    def post(self, request, serializer, **kwargs):
        agent_exchange(
            "project_load",
            dataset_path=TERRA_PATH.datasets,
            source=serializer.validated_data.get("value"),
            target=PROJECT_PATH.base,
        )
        return BaseResponseSuccess()


class LoadProgressAPIView(BaseAPIView):
    @decorators.progress_error(PROJECT_LOAD_NAME)
    def post(self, request, progress, **kwargs):
        if progress.finished:
            progress.percent = 0
            progress.message = ""
            request.project.load()
        return BaseResponseSuccess(progress.native())


class DeleteAPIView(BaseAPIView):
    @decorators.serialize_data(DeleteSerializer)
    def post(self, request, serializer, **kwargs):
        project = ProjectPathData(path=serializer.validated_data.get("path"))
        project.path.unlink()
        return BaseResponseSuccess()
