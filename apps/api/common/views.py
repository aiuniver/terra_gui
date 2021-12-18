from terra_ai.settings import TERRA_PATH

from apps.api import decorators
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.common.serializers import ValidateDatasetModelSerializer
from apps.api.logging import logs_catcher


class LogsAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(logs_catcher.logs)


class ValidateDatasetModelAPIView(BaseAPIView):
    @decorators.serialize_data(ValidateDatasetModelSerializer)
    def post(self, request, serializer, **kwargs):
        dataset_load = serializer.validated_data.get("dataset")
        model_load = serializer.validated_data.get("model")

        dataset = None
        model = None

        if dataset_load:
            datasets = self.terra_exchange("datasets_info", path=TERRA_PATH.datasets)
            dataset = datasets.get(dataset_load.get("group")).datasets.get(
                dataset_load.get("alias")
            )
            model = request.project.model

        if model_load:
            model = self.terra_exchange("model_get", value=model_load.get("value"))
            dataset = request.project.dataset

        if not dataset or not len(model.layers):
            validated = True
        else:
            validated = len(dataset.inputs.keys()) == len(model.inputs) and len(
                dataset.outputs.keys()
            ) == len(model.outputs)

        message = None
        if not validated:
            if dataset_load:
                message = "Несоответствие количества входных/выходных слоев датасета и редактируемой модели. Хотите сбросить модель?"
            elif model_load:
                message = "Несоответствие количества входных/выходных слоев датасета и редактируемой модели. Хотите сбросить датасет?"
            else:
                message = "Undefined message"

        return BaseResponseSuccess(message)
