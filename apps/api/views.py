import json

from django.conf import settings

from apps.plugins.frontend import presets
from apps.plugins.frontend.base import Field
from apps.plugins.frontend.defaults import DefaultsData
from apps.plugins.frontend.presets.defaults import (
    TrainingLosses,
    TrainingMetrics,
    TrainingLossSelect,
    TrainingMetricSelect,
    TrainingClassesQuantitySelect,
)

from .base import BaseAPIView, BaseResponseSuccess


class NotFoundAPIView(BaseAPIView):
    pass


class ConfigAPIView(BaseAPIView):
    def _get_training_outputs(self, data: list) -> dict:
        outputs = {}
        for layer in data:
            losses_data = {**TrainingLossSelect}
            losses_list = list(
                map(
                    lambda item: {"label": item.name, "value": item.value},
                    TrainingLosses.get(layer.task),
                )
            )
            losses_data.update(
                {
                    "name": losses_data.get("name") % layer.id,
                    "parse": losses_data.get("parse") % layer.id,
                    "value": losses_list[0].get("label"),
                    "list": losses_list,
                }
            )
            metrics_data = {**TrainingMetricSelect}
            metrics_list = list(
                map(
                    lambda item: {"label": item.name, "value": item.value},
                    TrainingMetrics.get(layer.task),
                )
            )
            metrics_data.update(
                {
                    "name": metrics_data.get("name") % layer.id,
                    "parse": metrics_data.get("parse") % layer.id,
                    "value": metrics_list[0].get("label"),
                    "list": metrics_list,
                }
            )
            classes_quantity_data = {**TrainingClassesQuantitySelect}
            classes_quantity_data.update(
                {
                    "name": classes_quantity_data.get("name") % layer.id,
                    "parse": classes_quantity_data.get("parse") % layer.id,
                    "value": layer.num_classes,
                }
            )
            outputs.update(
                {
                    layer.id: {
                        "name": f"Слой «{layer.name}»",
                        "classes_quantity": layer.num_classes,
                        "fields": [
                            Field(**losses_data),
                            Field(**metrics_data),
                            Field(**classes_quantity_data),
                        ],
                    }
                }
            )
        return outputs

    def post(self, request, **kwargs):
        defaults = DefaultsData(**presets.defaults.Defaults)
        defaults.training.base.outputs.fields = self._get_training_outputs(
            request.project.model.outputs
        )
        return BaseResponseSuccess(
            {
                "defaults": json.loads(defaults.json()),
                "project": json.loads(request.project.json()),
                "user": {
                    "login": settings.USER_LOGIN,
                    "first_name": settings.USER_NAME,
                    "last_name": settings.USER_LASTNAME,
                },
            }
        )
