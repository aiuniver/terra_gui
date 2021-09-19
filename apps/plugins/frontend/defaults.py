from typing import List, Dict, Optional, Union

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.layer import LayersList
from terra_ai.data.modeling.model import ModelDetailsData

from apps.plugins.frontend.presets.defaults import (
    TrainingLosses,
    TrainingMetrics,
    TrainingLossSelect,
    TrainingMetricSelect,
    TrainingClassesQuantitySelect,
)

from .base import Field


class DefaultsDatasetsCreationData(BaseMixinData):
    column_processing: List[Field]
    input: List[Field]
    output: List[Field]


class DefaultsDatasetsData(BaseMixinData):
    creation: DefaultsDatasetsCreationData


class DefaultsModelingData(BaseMixinData):
    layer_form: List[Field]
    layers_types: dict


class DefaultsTrainingBaseGroupData(BaseMixinData):
    name: Optional[str]
    collapsable: bool = False
    collapsed: bool = False
    fields: Union[List[Field], Dict[str, List[Field]]]


class DefaultsTrainingBaseData(BaseMixinData):
    main: DefaultsTrainingBaseGroupData
    fit: DefaultsTrainingBaseGroupData
    optimizer: DefaultsTrainingBaseGroupData
    outputs: DefaultsTrainingBaseGroupData
    checkpoint: DefaultsTrainingBaseGroupData


class DefaultsTrainingData(BaseMixinData):
    base: DefaultsTrainingBaseData


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData

    def __update_training_outputs(self, layers: LayersList):
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
        self.training.base.outputs.fields = outputs

    def __update_training_checkpoint(self, layers: LayersList):
        layers_choice = []
        for layer in layers:
            layers_choice.append(
                {
                    "value": layer.id,
                    "label": f"Слой «{layer.name}»",
                }
            )
        if layers_choice:
            for index, item in enumerate(self.training.base.checkpoint.fields):
                if item.name == "architecture_parameters_checkpoint_layer":
                    field_data = item.native()
                    field_data.update(
                        {
                            "value": str(layers_choice[0].get("value")),
                            "list": layers_choice,
                        }
                    )
                    self.training.base.checkpoint.fields[index] = Field(**field_data)
                    break

    def update_by_model(self, model: ModelDetailsData):
        self.__update_training_outputs(model.outputs)
        self.__update_training_checkpoint(model.outputs)
