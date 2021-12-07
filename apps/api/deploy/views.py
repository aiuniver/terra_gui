import json
import hashlib

from pathlib import Path

from django.conf import settings

from terra_ai.settings import TERRA_PATH, PROJECT_PATH, DEPLOY_PATH
from terra_ai.agent import agent_exchange
from terra_ai.deploy.prepare_deploy import DeployCreator
from terra_ai.data.datasets.dataset import DatasetInfo, DatasetLoadData
from terra_ai.data.deploy.tasks import DeployPageData
from terra_ai.data.deploy.extra import DeployTypePageChoice

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.deploy.serializers import (
    GetSerializer,
    ReloadSerializer,
    UploadSerializer,
)


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(GetSerializer)
    def post(self, request, serializer, **kwargs):
        page = DeployPageData(**serializer.validated_data)
        datasets = []
        if page.type == DeployTypePageChoice.model:
            _path = Path(PROJECT_PATH.training, page.name, "model", "dataset.json")
            if not _path.is_file():
                _path = Path(
                    PROJECT_PATH.training, page.name, "model", "dataset", "config.json"
                )
            with open(_path) as dataset_ref:
                dataset_config = json.load(dataset_ref)
                datasets.append(
                    DatasetLoadData(path=TERRA_PATH.datasets, **dataset_config)
                )
        agent_exchange("deploy_get", datasets=datasets, page=page)
        return BaseResponseSuccess()


class GetProgressAPIView(BaseAPIView):
    @decorators.progress_error("deploy_get")
    def post(self, request, progress, **kwargs):
        if progress.success:
            if progress.finished:
                progress.percent = 0
                progress.message = ""
                datasets = progress.data.get("datasets")
                dataset_data = datasets[0].native() if len(datasets) else None
                dataset = DatasetInfo(**dataset_data).dataset if dataset_data else None
                request.project.deploy = DeployCreator().get_deploy(
                    dataset=dataset,
                    training_path=PROJECT_PATH.training,
                    deploy_path=DEPLOY_PATH,
                    page=progress.data.get("kwargs", {}).get("page").native(),
                )
                progress.data = request.project.deploy.presets
        return BaseResponseSuccess(progress.native())


class ReloadAPIView(BaseAPIView):
    @decorators.serialize_data(ReloadSerializer)
    def post(self, request, serializer, **kwargs):
        if request.project.deploy:
            request.project.deploy.data.reload(serializer.validated_data)
        request.project.save_config()
        return BaseResponseSuccess(request.project.deploy.presets)


class UploadAPIView(BaseAPIView):
    @decorators.serialize_data(UploadSerializer)
    def post(self, request, serializer, **kwargs):
        sec = serializer.validated_data.get("sec")
        agent_exchange(
            "deploy_upload",
            **{
                "source": DEPLOY_PATH,
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
    @decorators.progress_error("deploy_upload")
    def post(self, request, progress, **kwargs):
        return BaseResponseSuccess(progress.native())
