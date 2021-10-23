import re
import base64

from tempfile import NamedTemporaryFile
from pydantic import ValidationError
from transliterate import slugify
from collections.abc import MutableMapping

from apps.plugins.project import data_path, Project
from apps.plugins.project import exceptions as project_exceptions

from terra_ai.agent import agent_exchange
from terra_ai.agent import exceptions as agent_exceptions
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.modeling.extra import LayerGroupChoice
from terra_ai.exceptions.base import TerraBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import (
    ModelGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
    CreateSerializer,
    DatatypeSerializer,
)
from .utils import autocrop_image_square


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
        try:
            model = agent_exchange(
                "model_get", value=serializer.validated_data.get("value")
            )
            return BaseResponseSuccess(model.native())
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            model = agent_exchange(
                "model_get", value=serializer.validated_data.get("value")
            )
            request.project.clear_training()
            reset_dataset = serializer.validated_data.get("reset_dataset")
            if reset_dataset:
                request.project.set_dataset()
            request.project.set_model(model)
            return BaseResponseSuccess(
                request.project.model.native(), save_project=True
            )
        except project_exceptions.ProjectException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            return BaseResponseSuccess(
                agent_exchange("models", path=data_path.modeling).native()
            )
        except (agent_exceptions.ExchangeBaseException, TerraBaseException) as error:
            return BaseResponseErrorGeneral(str(error))


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_model()
        return BaseResponseSuccess(save_project=True)


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
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
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            answer = BaseResponseErrorFields(error)
            buff_error = flatten_dict(answer.data["error"])
            error = dict(
                (key[: len(key) - 1], value) for (key, value) in buff_error.items()
            )
            answer.data["error"] = error
            return answer


class ValidateAPIView(BaseAPIView):
    def _reset_layers_shape(
        self, dataset: DatasetData, model: ModelDetailsData
    ) -> ModelDetailsData:
        for layer in model.middles:
            layer.shape.input = []
            layer.shape.output = []
        for index, layer in enumerate(model.inputs):
            layer.shape.output = []
            if dataset:
                layer.shape = dataset.model.inputs[index].shape
        for index, layer in enumerate(model.outputs):
            layer.shape.input = []
            if dataset:
                layer.shape = dataset.model.outputs[index].shape
            else:
                layer.shape.output = []
        return model

    def post(self, request, **kwargs):
        try:
            source_model = self._reset_layers_shape(
                request.project.dataset, request.project.model
            )
            validated_model, errors = agent_exchange(
                "model_validate",
                model=source_model,
            )
            request.project.set_model(validated_model)
            return BaseResponseSuccess(errors, save_project=True)
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class PreviewAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PreviewSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        filepath = NamedTemporaryFile(suffix=".png")  # Add for Win ,delete=False
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
        try:
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
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            agent_exchange("model_delete", path=request.data.get("path"))
            return BaseResponseSuccess()
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))


class DatatypeAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = DatatypeSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        return BaseResponseSuccess(request.project.model.native())
