from terra_ai.agent import agent_exchange
from terra_ai.data.projects.project import ProjectPathData

from apps.plugins.project import project_path, data_path

from apps.api.base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields

from . import serializers


class NameAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.NameSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.set_name(serializer.validated_data.get("name"))
        return BaseResponseSuccess()


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.create()
        return BaseResponseSuccess()


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.set_name(serializer.validated_data.get("name"))
        request.project.save(serializer.validated_data.get("overwrite"))
        return BaseResponseSuccess()


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("projects_info", path=data_path.projects).native()
        )


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.LoadSerializer(data=request.data)
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
        serializer = serializers.DeleteSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        project = ProjectPathData(path=serializer.validated_data.get("path"))
        project.path.unlink()
        return BaseResponseSuccess()
