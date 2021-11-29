import json
import hashlib

from pathlib import Path

from django.conf import settings

from terra_ai import settings as terra_ai_settings
from terra_ai.agent import agent_exchange
from terra_ai.deploy.prepare_deploy import DeployCreator
from terra_ai.data.datasets.dataset import DatasetInfo, DatasetLoadData
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
        print(1)
        serializer = serializers.GetSerializer(data=request.data)
        print(2)
        if not serializer.is_valid():
            print(3)
            return BaseResponseErrorFields(serializer.errors)
        print(4)
        page = DeployPageData(**serializer.validated_data)
        print(5)
        datasets = []
        if page.type == DeployTypePageChoice.model:
            print(6)
            _path = Path(project_path.training, page.name, "model", "dataset.json")
            print(7)
            if not _path.is_file():
                print(8)
                _path = Path(
                    project_path.training, page.name, "model", "dataset", "config.json"
                )
                print(9)
            print(10)
            with open(_path) as dataset_ref:
                print(11)
                dataset_config = json.load(dataset_ref)
                print(12)
                datasets.append(
                    DatasetLoadData(path=data_path.datasets, **dataset_config)
                )
                print(13)
        print(14)
        agent_exchange("deploy_get", datasets=datasets, page=page)
        print(15)
        return BaseResponseSuccess()


class GetProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("deploy_get_progress")
        if progress.success:
            if progress.finished:
                progress.percent = 0
                progress.message = ""
                datasets = progress.data.get("datasets")
                dataset_data = datasets[0].native() if len(datasets) else None
                dataset = DatasetInfo(**dataset_data).dataset if dataset_data else None
                request.project.deploy = DeployCreator().get_deploy(
                    dataset=dataset,
                    training_path=project_path.training,
                    deploy_path=terra_ai_settings.DEPLOY_PATH,
                    page=progress.data.get("kwargs", {}).get("page").native(),
                )
                progress.data = request.project.deploy.presets
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
