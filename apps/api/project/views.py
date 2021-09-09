from terra_ai.agent import agent_exchange
from terra_ai.agent import exceptions as agent_exceptions

from apps.plugins.project import project_path, data_path

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import NameSerializer, SaveSerializer


class NameAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = NameSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.name = serializer.validated_data.get("name")
        return BaseResponseSuccess(update_project=True)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("projects_info", path=data_path.projects).native()
        )


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess()


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.reset()
        return BaseResponseSuccess(update_project=True)


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


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess()
