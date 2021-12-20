import base64

from tempfile import NamedTemporaryFile

from terra_ai.settings import TERRA_PATH, PROJECT_PATH
from terra_ai.data.datasets.dataset import DatasetInfo
from terra_ai.data.cascades.extra import BlockGroupChoice

from apps.api import decorators
from apps.api.utils import autocrop_image_square
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.cascades.serializers import (
    CascadeGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
    StartSerializer,
    SaveSerializer,
)


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(CascadeGetSerializer)
    def post(self, request, serializer, **kwargs):
        cascade = self.terra_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange("cascades_info", path=PROJECT_PATH.cascades).native()
        )


class LoadAPIView(BaseAPIView):
    @decorators.serialize_data(CascadeGetSerializer)
    def post(self, request, serializer, **kwargs):
        request.project.cascade = self.terra_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(request.project.cascade.native())


class UpdateAPIView(BaseAPIView):
    @decorators.serialize_data(UpdateSerializer)
    def post(self, request, serializer, **kwargs):
        cascade = request.project.cascade
        data = serializer.validated_data
        cascade_data = cascade.native()
        cascade_data.update(data)
        cascade = self.terra_exchange("cascade_update", cascade=cascade_data)
        request.project.set_cascade(cascade)
        return BaseResponseSuccess({"blocks": cascade.blocks.native()})


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_cascade()
        return BaseResponseSuccess(request.project.cascade.native())


class ValidateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange(
                "cascade_validate",
                path=PROJECT_PATH.training,
                cascade=request.project.cascade,
            )
        )


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = StartSerializer(data={"sources": request.data})
        serializer.is_valid(raise_exception=True)
        self.terra_exchange(
            "cascade_start",
            training_path=PROJECT_PATH.training,
            datasets_path=TERRA_PATH.datasets,
            sources=serializer.validated_data.get("sources"),
            cascade=request.project.cascade,
        )
        return BaseResponseSuccess()


class StartProgressAPIView(BaseAPIView):
    @decorators.progress_error("cascade_start")
    def post(self, request, progress, **kwargs):
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
            self.terra_exchange(
                "cascade_execute",
                sources=sources,
                cascade=request.project.cascade,
                training_path=PROJECT_PATH.training,
            )
            progress.message = ""
            progress.percent = 0
            progress.data = None
        return BaseResponseSuccess(progress.native())


class SaveAPIView(BaseAPIView):
    @decorators.serialize_data(SaveSerializer)
    def post(self, request, serializer, **kwargs):
        request.project.cascade.save(
            path=PROJECT_PATH.cascades, **serializer.validated_data
        )
        return BaseResponseSuccess()


class PreviewAPIView(BaseAPIView):
    @decorators.serialize_data(PreviewSerializer)
    def post(self, request, serializer, **kwargs):
        filepath = NamedTemporaryFile(suffix=".png")
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))


class DatasetsAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        datasets_list = self.terra_exchange(
            "datasets_info", path=TERRA_PATH.datasets
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
