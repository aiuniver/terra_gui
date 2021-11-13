import hashlib

from pathlib import Path

from django.conf import settings

from terra_ai.agent import agent_exchange

from apps.api.base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)

from . import serializers


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.GetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.set_deploy(serializer.validated_data)
        return BaseResponseSuccess(
            request.project.deploy.presets if request.project.deploy else None
        )


class ReloadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.ReloadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        if request.project.deploy:
            request.project.deploy.data.reload(serializer.validated_data)
        request.project.save_config()
        return BaseResponseSuccess(request.project.deploy.presets)


class UploadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.UploadSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        sec = serializer.validated_data.get("sec")
        agent_exchange(
            "deploy_upload",
            **{
                "source": Path(request.project.training.path, "deploy"),
                "stage": 1,
                "deploy": serializer.validated_data.get("deploy"),
                "env": "v1",
                "user": {
                    "login": settings.USER_LOGIN,
                    "name": settings.USER_NAME,
                    "lastname": settings.USER_LASTNAME,
                    "sec": hashlib.md5(sec.encode("utf-8")).hexdigest() if sec else "",
                },
                "project": {
                    "name": request.project.name,
                },
                "task": request.project.deploy.type.demo,
                "replace": serializer.validated_data.get("replace"),
            }
        )
        return BaseResponseSuccess()


class UploadProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("deploy_upload_progress")
        if progress.success:
            return BaseResponseSuccess(data=progress.native())
        else:
            return BaseResponseErrorGeneral(progress.error, data=progress.native())
