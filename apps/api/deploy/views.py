from pydantic import ValidationError

from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import UploadSerializer


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        print(request.project)
        return BaseResponseSuccess()


class UploadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UploadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            stage = agent_exchange(
                "deploy_upload",
                **{
                    "stage": 1,
                    "user": {
                        "login": "bl146u",
                        "name": "Юрий",
                        "lastname": "Максимов",
                    },
                    "project_name": "NoName",
                    "url": serializer.validated_data.get("url"),
                    "replace": serializer.validated_data.get("replace"),
                    "filename": "asdasdas.zip",
                    "filesize": 38456,
                }
            )
            return BaseResponseSuccess(stage.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)
