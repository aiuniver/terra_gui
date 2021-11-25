import re
import base64

from tempfile import NamedTemporaryFile
from transliterate import slugify

from terra_ai.agent import agent_exchange
from terra_ai.data.modeling.extra import LayerGroupChoice
from terra_ai.data.modeling.model import ModelDetailsData

from apps.plugins.project import data_path

from apps.api import utils
from apps.api.base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields

from . import serializers


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        model = agent_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(model.native())


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        model = agent_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        request.project.set_model(model, serializer.validated_data.get("reset_dataset"))
        return BaseResponseSuccess(request.project.model.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=data_path.modeling).native()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_model()
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
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
        model = agent_exchange("model_update", model=model_data)
        request.project.set_model(model)
        return BaseResponseSuccess()


class ValidateAPIView(BaseAPIView):
    @staticmethod
    def _reset_layers_shape(model: ModelDetailsData):
        for layer in model.middles:
            layer.shape.input = []
            layer.shape.output = []
        for index, layer in enumerate(model.inputs):
            layer.shape.output = []
        for index, layer in enumerate(model.outputs):
            layer.shape.input = []

    def post(self, request, **kwargs):
        self._reset_layers_shape(request.project.model)
        errors = agent_exchange("model_validate", model=request.project.model)
        request.project.save_config()
        return BaseResponseSuccess(errors)


class PreviewAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.PreviewSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        filepath = NamedTemporaryFile(suffix=".png")  # Add for Win ,delete=False
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        utils.autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
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
            data=agent_exchange(
                "model_create",
                model=model.native(),
                path=str(data_path.modeling),
                overwrite=serializer.validated_data.get("overwrite"),
            )
        )


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("model_delete", path=request.data.get("path"))
        return BaseResponseSuccess()


class DatatypeAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = serializers.DatatypeSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        source_id = serializer.validated_data.get("source")
        target_id = serializer.validated_data.get("target")
        if source_id != target_id:
            request.project.model.reindex(source_id=source_id, target_id=target_id)
            if request.project.dataset:
                request.project.model.update_layers(request.project.dataset)
            request.project.save_config()
        return BaseResponseSuccess(request.project.model.native())
