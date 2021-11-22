import json
import hashlib

from pathlib import Path

from django.conf import settings

from terra_ai import settings as terra_ai_settings
from terra_ai.agent import agent_exchange
from terra_ai.data.datasets.dataset import DatasetLoadData
from terra_ai.data.deploy.tasks import DeployPageData
from terra_ai.data.deploy.extra import DeployTypePageChoice

from apps.api.base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from apps.plugins.project import project_path, data_path

from . import serializers


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.GetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        page = DeployPageData(**serializer.validated_data)
        datasets = []
        if page.type == DeployTypePageChoice.model:
            with open(
                Path(project_path.training, page.name, "model", "dataset.json")
            ) as dataset_ref:
                dataset_config = json.load(dataset_ref)
                datasets.append(
                    DatasetLoadData(path=data_path.datasets, **dataset_config)
                )
        elif page.type == DeployTypePageChoice.cascade:
            print(page)
        agent_exchange("deploy_get", datasets=datasets, page=page)
        return BaseResponseSuccess()


class GetProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("deploy_get")

        # request.project.set_deploy(
        #     dataset=request.project.dataset, page=serializer.validated_data
        # )
        # return BaseResponseSuccess(
        #     request.project.deploy.presets if request.project.deploy else None
        # )

        if progress.success:
            return BaseResponseSuccess(data=progress.native())
        else:
            return BaseResponseErrorGeneral(progress.error, data=progress.native())


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
                "source": terra_ai_settings.DEPLOY_PATH,
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
