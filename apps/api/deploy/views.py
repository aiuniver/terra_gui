import hashlib

from pathlib import Path
from pydantic import ValidationError

from django.conf import settings

from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import PrepareSerializer


class PrepareAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PrepareSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            # Подготовить zip-файл
            # ...
            filepath = Path("/tmp/aaa.zip")
            sec = serializer.validated_data.get("sec")
            stage = agent_exchange(
                "deploy_upload",
                **{
                    "stage": 1,
                    "deploy": serializer.validated_data.get("deploy"),
                    "user": {
                        "login": settings.USER_LOGIN,
                        "name": settings.USER_NAME,
                        "lastname": settings.USER_LASTNAME,
                        "sec": hashlib.md5(sec.encode("utf-8")).hexdigest()
                        if sec
                        else "",
                    },
                    "project": {
                        "name": request.project.name,
                    },
                    "task": "text_classification",
                    "replace": serializer.validated_data.get("replace"),
                    "file": {
                        "path": filepath,
                    },
                }
            )
            return BaseResponseSuccess(stage.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class UploadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PrepareSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            # Подготовить zip-файл
            # ...
            filepath = Path("/tmp/aaa.zip")
            stage = agent_exchange(
                "deploy_upload",
                **{
                    "stage": 1,
                    "deploy": serializer.validated_data.get("deploy"),
                    "user": {
                        "login": settings.USER_LOGIN,
                        "name": settings.USER_NAME,
                        "lastname": settings.USER_LASTNAME,
                    },
                    "project_name": request.project.name,
                    "replace": serializer.validated_data.get("replace"),
                    "file": {
                        "path": filepath,
                    },
                }
            )
            return BaseResponseSuccess(stage.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)
