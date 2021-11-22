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
            # datasets += list(
            #     map(
            #         lambda item: DatasetLoadData(path=datasets_path, **dict(item)),
            #         sources.values(),
            #     )
            # )
            # for block in cascade.blocks:
            #     if block.group == BlockGroupChoice.Model:
            #         _path = Path(
            #             training_path,
            #             block.parameters.main.path,
            #             "model",
            #             "dataset.json",
            #         )
            #         with open(_path) as config_ref:
            #             data = json.load(config_ref)
            #             datasets.append(
            #                 DatasetLoadData(
            #                     path=datasets_path,
            #                     alias=data.get("alias"),
            #                     group=data.get("group"),
            #                 )
            #             )
            # datasets_loading.multiload("cascade_start", datasets, sources=sources)
            pass

        agent_exchange("deploy_get", datasets=datasets, page=page)
        return BaseResponseSuccess()


class GetProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("deploy_get_progress")
        if progress.success:
            if progress.finished:
                progress.percent = 0
                progress.message = ""
                dataset_data = progress.data.get("datasets")[0]
                request.project.deploy = DeployCreator().get_deploy(
                    dataset=DatasetInfo(**dataset_data.native()).dataset,
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
