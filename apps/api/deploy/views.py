import hashlib

from pathlib import Path
from pydantic import ValidationError

from django.conf import settings

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException
from terra_ai.exceptions.base import TerraBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import UploadSerializer


class UploadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UploadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            sec = serializer.validated_data.get("sec")
            agent_exchange(
                "deploy_upload",
                **{
                    "source": Path("./TerraAI/tmp"),
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
                    "task": "image_classification",
                    "replace": serializer.validated_data.get("replace"),
                }
            )
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
        except (TerraBaseException, ExchangeBaseException) as error:
            return BaseResponseErrorGeneral(str(error))


class UploadProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            progress = agent_exchange("deploy_upload_progress")
            if progress.success:
                return BaseResponseSuccess(data=progress.native())
            else:
                return BaseResponseErrorGeneral(progress.error, data=progress.native())
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
