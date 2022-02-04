import re
import base64

from tempfile import NamedTemporaryFile
from transliterate import slugify
from pydantic.error_wrappers import ValidationError

from terra_ai.settings import TERRA_PATH
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.extra import LayerGroupChoice
from terra_ai.data.modeling.model import ModelDetailsData

from apps.api import decorators
from apps.api.utils import autocrop_image_square
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.modeling.serializers import (
    ModelGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
    CreateSerializer,
    DatatypeSerializer,
)


class GetAPIView(BaseAPIView):
    @decorators.serialize_data(ModelGetSerializer)
    def post(self, request, serializer, **kwargs):
        model = self.terra_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(model.native())


class LoadAPIView(BaseAPIView):
    @decorators.serialize_data(ModelGetSerializer)
    def post(self, request, serializer, **kwargs):
        model = self.terra_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        request.project.set_model(model, serializer.validated_data.get("reset_dataset"))
        return BaseResponseSuccess(request.project.model.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            self.terra_exchange("models", path=TERRA_PATH.modeling).native()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_model()
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    @decorators.serialize_data(UpdateSerializer)
    def post(self, request, serializer, **kwargs):
        model = request.project.model
        data = serializer.validated_data
        for item in data.get("layers"):
            layer = model.layers.get(item.get("id"))
            if layer:
                shape = layer.shape.native()
                if (
                    item.get("group") == LayerGroupChoice.input
                    and request.project.dataset is None
                ):
                    shape["input"] = item.get("shape", {}).get("input", [])
                item["shape"] = shape
            else:
                if (
                    item.get("group") == LayerGroupChoice.input
                    and request.project.dataset is None
                ):
                    item["shape"] = {"input": item.get("shape", {}).get("input", [])}
                else:
                    del item["shape"]
        model_data = model.native()
        model_data.update(data)
        errors = {}
        try:
            model = self.terra_exchange("model_update", model=model_data)
            request.project.set_model(model)
        except ValidationError as exc:
            errors = self._errors_processing(exc)
        return BaseResponseSuccess(errors)

    def _errors_processing(self, exc: ValidationError) -> dict:
        errors = {}
        for error in exc.errors():
            loc = error.get("loc")
            name = loc[0]
            tail = loc[1:]
            if len(tail):
                name += f'[{"][".join(tail)}]'
            errors.update({name: str(error.get("msg"))})
        return errors


class ValidateAPIView(BaseAPIView):
    @staticmethod
    def _reset_layers_shape(model: ModelDetailsData, dataset_model: DatasetData = None):
        for layer in model.middles:
            layer.shape.input = []
            layer.shape.output = []
        for index, layer in enumerate(model.inputs):
            layer.shape.output = []
            layer.shape.input = (
                dataset_model.inputs.get(layer.id).shape.input
                if dataset_model
                else layer.shape.input
            )
        for index, layer in enumerate(model.outputs):
            layer.shape.input = []
            layer.shape.output = (
                dataset_model.outputs.get(layer.id).shape.output
                if dataset_model
                else []
            )

    def post(self, request, **kwargs):
        self._reset_layers_shape(
            request.project.model,
            request.project.dataset.model if request.project.dataset else None,
        )
        errors = self.terra_exchange(
            "model_validate",
            model=request.project.model,
            dataset_data=request.project.dataset
            if request.project.dataset
            else None,
        )
        request.project.save_config()
        return BaseResponseSuccess(errors)


class PreviewAPIView(BaseAPIView):
    @decorators.serialize_data(PreviewSerializer)
    def post(self, request, serializer, **kwargs):
        filepath = NamedTemporaryFile(suffix=".png", delete=False)
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))


class CreateAPIView(BaseAPIView):
    @decorators.serialize_data(CreateSerializer)
    def post(self, request, serializer, **kwargs):
        model_data = request.project.model.native()
        model_data.update(
            {
                "name": serializer.validated_data.get("name"),
                "alias": re.sub(
                    r"([\-]+)",
                    "_",
                    slugify(serializer.validated_data.get("name"), language_code="ru"),
                ),
                "image": serializer.validated_data.get("preview"),
            }
        )
        model = ModelDetailsData(**model_data)
        return BaseResponseSuccess(
            self.terra_exchange(
                "model_create",
                model=model.native(),
                path=str(TERRA_PATH.modeling),
                overwrite=serializer.validated_data.get("overwrite"),
            )
        )


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        self.terra_exchange("model_delete", path=request.data.get("path"))
        return BaseResponseSuccess()


class DatatypeAPIView(BaseAPIView):
    @decorators.serialize_data(DatatypeSerializer)
    def post(self, request, serializer, **kwargs):
        source_id = serializer.validated_data.get("source")
        target_id = serializer.validated_data.get("target")
        if source_id != target_id:
            request.project.model.reindex(source_id=source_id, target_id=target_id)
            if request.project.dataset:
                request.project.model.update_layers(request.project.dataset)
            request.project.save_config()
        return BaseResponseSuccess(request.project.model.native())
