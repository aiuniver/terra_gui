import json

from django.conf import settings

from terra_ai.data.modeling.layer import LayersList

from apps.plugins.frontend import presets
from apps.plugins.frontend.base import Field
from apps.plugins.frontend.defaults import DefaultsData, DefaultsTrainingBaseGroupData
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
    def _get_training_outputs(self, layers: LayersList) -> dict:
        outputs = {}
        for layer in layers:
            losses_data = {**TrainingLossSelect}
            losses_list = list(
                map(
                    lambda item: {"label": item.name, "value": item.value},
                    TrainingLosses.get(layer.task, []),
                )
            )
            losses_data.update(
                {
                    "name": losses_data.get("name") % layer.id,
                    "parse": losses_data.get("parse") % layer.id,
                    "value": losses_list[0].get("label") if losses_list else "",
                    "list": losses_list,
                }
            )
            metrics_data = {**TrainingMetricSelect}
            metrics_list = list(
                map(
                    lambda item: {"label": item.name, "value": item.value},
                    TrainingMetrics.get(layer.task, []),
                )
            )
            metrics_data.update(
                {
                    "name": metrics_data.get("name") % layer.id,
                    "parse": metrics_data.get("parse") % layer.id,
                    "value": [metrics_list[0].get("label")] if metrics_list else [],
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

    def _get_training_checkpoint(
        self, checkpoint: DefaultsTrainingBaseGroupData, layers: LayersList
    ):
        layers_choice = []
        for layer in layers:
            layers_choice.append(
                {
                    "value": layer.id,
                    "label": f"Слой «{layer.name}»",
                }
            )
        if layers_choice:
            for index, item in enumerate(checkpoint.fields):
                if item.name == "architecture_parameters_checkpoint_layer":
                    field_data = item.native()
                    field_data.update(
                        {
                            "value": str(layers_choice[0].get("value")),
                            "list": layers_choice,
                        }
                    )
                    checkpoint.fields[index] = Field(**field_data)
                    break
        return checkpoint

    def post(self, request, **kwargs):
        defaults = DefaultsData(**presets.defaults.Defaults)
        defaults.training.base.outputs.fields = self._get_training_outputs(
            request.project.model.outputs
        )
        defaults.training.base.checkpoint = self._get_training_checkpoint(
            defaults.training.base.checkpoint, request.project.model.outputs
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
