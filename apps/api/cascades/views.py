import base64

from tempfile import NamedTemporaryFile

from terra_ai.agent import agent_exchange
from terra_ai.data.datasets.dataset import DatasetInfo
from terra_ai.data.cascades.extra import BlockGroupChoice

from apps.api import utils
from apps.api.cascades.serializers import (
    CascadeGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
    StartSerializer,
    SaveSerializer,
)
from apps.plugins.project import project_path, data_path

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
)


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("cascades_info", path=project_path.cascades).native()
        )


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = request.project.cascade
        data = serializer.validated_data
        cascade_data = cascade.native()
        cascade_data.update(data)
        cascade = agent_exchange("cascade_update", cascade=cascade_data)
        request.project.set_cascade(cascade)
        return BaseResponseSuccess({"blocks": cascade.blocks.native()})


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_cascade()
        return BaseResponseSuccess(request.project.cascade.native())


class ValidateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange(
                "cascade_validate",
                path=project_path.training,
                cascade=request.project.cascade,
            )
        )


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = StartSerializer(data={"sources": request.data})
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        agent_exchange(
            "cascade_start",
            training_path=project_path.training,
            datasets_path=data_path.datasets,
            sources=serializer.validated_data.get("sources"),
            cascade=request.project.cascade,
        )
        return BaseResponseSuccess()


class StartProgressAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        progress = agent_exchange("cascade_start_progress")
        if progress.finished:
            sources_data = progress.data.get("kwargs", {}).get("sources", {})
            dataset_sources = progress.data.get("datasets", [])
            sources = {}
            for block in request.project.cascade.blocks:
                if block.group != BlockGroupChoice.InputData:
                    continue
                source_data = sources_data.get(str(block.id))
                if not source_data:
                    continue
                datasets_source = list(
                    filter(
                        lambda item: item.alias == source_data.get("alias")
                        and item.group == source_data.get("group"),
                        dataset_sources,
                    )
                )
                if not len(datasets_source):
                    continue
                sources.update(
                    {
                        block.id: DatasetInfo(
                            alias=datasets_source[0].alias,
                            group=datasets_source[0].group,
                        ).dataset.sources
                    }
                )
            agent_exchange(
                "cascade_execute",
                sources=sources,
                cascade=request.project.cascade,
                training_path=project_path.training,
            )
            progress.message = ""
            progress.percent = 0
            progress.data = None
        return BaseResponseSuccess(progress.native())


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.cascade.save(
            path=project_path.cascades, **serializer.validated_data
        )
        return BaseResponseSuccess()


class PreviewAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PreviewSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        filepath = NamedTemporaryFile(suffix=".png")  # Add for Win ,delete=False
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        utils.autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))


class DatasetsAPIView(BaseAPIView):
    @staticmethod
    def post(request, **kwargs):
        datasets_list = agent_exchange(
            "datasets_info", path=data_path.datasets
        ).native()
        response = []

        for datasets in datasets_list:
            for dataset in datasets.get("datasets", []):
                response.append(
                    {
                        "label": f'{dataset.get("group", "")}: {dataset.get("name", "")}',
                        "alias": dataset.get("alias", ""),
                        "group": dataset.get("group", ""),
                    }
                )

        return BaseResponseSuccess(response)
