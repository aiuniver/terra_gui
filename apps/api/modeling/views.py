import base64
import re
from collections.abc import MutableMapping
from tempfile import NamedTemporaryFile

from transliterate import slugify

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange
from terra_ai.data.modeling.extra import LayerGroupChoice
from terra_ai.data.modeling.model import ModelDetailsData
from .serializers import (
    ModelGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
    CreateSerializer,
    DatatypeSerializer,
)
from .utils import autocrop_image_square
from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
)


def flatten_dict(
    d: MutableMapping, parent_key: str = "[", sep: str = "]["
) -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{k}{sep}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        model = agent_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(model.native())


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        model = agent_exchange(
            "model_get", value=serializer.validated_data.get("value")
        )
        request.project.clear_training()
        reset_dataset = serializer.validated_data.get("reset_dataset")
        if reset_dataset:
            request.project.set_dataset()
        else:
            request.project.set_model(model)
        return BaseResponseSuccess(
            request.project.model.native(), save_project=True
        )


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=data_path.modeling).native()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_model()
        return BaseResponseSuccess(save_project=True)


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
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
                    item["shape"] = {
                        "input": item.get("shape", {}).get("input", [])
                    }
                else:
                    del item["shape"]
        model_data = model.native()
        model_data.update(data)
        model = agent_exchange("model_update", model=model_data)
        request.project.set_model(model)
        return BaseResponseSuccess(save_project=True)
        # except ValidationError as error:
        #     answer = BaseResponseErrorFields(error)
        #     buff_error = flatten_dict(answer.data["error"])
        #     error = dict(
        #         (key[: len(key) - 1], value) for (key, value) in buff_error.items()
        #     )
        #     answer.data["error"] = error
        #     return answer


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
        return BaseResponseSuccess(errors, save_project=True)


class PreviewAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PreviewSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        filepath = NamedTemporaryFile(suffix=".png",delete=False)  # Add for Win ,delete=False
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        model_data = request.project.model.native()
        model_data.update(
            {
                "name": serializer.validated_data.get("name"),
                "alias": re.sub(
                    r"([\-]+)",
                    "_",
                    slugify(
                        serializer.validated_data.get("name"), language_code="ru"
                    ),
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
        serializer = DatatypeSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        source_id = serializer.validated_data.get("source")
        target_id = serializer.validated_data.get("target")
        if source_id != target_id:
            request.project.model.switch_index(source_id=source_id, target_id=target_id)
            request.project.update_model_layers()
        return BaseResponseSuccess(request.project.model.native())
