from apps.plugins.project import data_path

from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import FailedGetModelException

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import ValidateDatasetModelSerializer


class ValidateDatasetModelAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ValidateDatasetModelSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)

        dataset_load = serializer.validated_data.get("dataset")
        model_load = serializer.validated_data.get("model")

        dataset = None
        model = None

        if dataset_load:
            datasets = agent_exchange("datasets_info", path=data_path.datasets)
            dataset = datasets.get(dataset_load.get("group")).datasets.get(
                dataset_load.get("alias")
            )
            model = request.project.model

        if model_load:
            try:
                model = agent_exchange("model_get", value=model_load.get("value"))
                dataset = request.project.dataset
            except FailedGetModelException as error:
                return BaseResponseErrorFields(error.args)

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
