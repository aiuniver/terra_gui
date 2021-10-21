import sys

from typing import List, Dict, Optional, Union, Any
from pydantic.main import ModelMetaclass

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.layer import LayersList
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.train import TrainData
from terra_ai.data.training.extra import ArchitectureChoice

from .presets.defaults.training import (
    TrainingTasksRelations,
    TrainingLossSelect,
    TrainingMetricSelect,
    TrainingClassesQuantitySelect,
    Architectures,
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


class YoloParametersGroup(BaseMixinData):
    train_lr_init: Field
    train_lr_end: Field
    yolo_iou_loss_thresh: Field
    train_warmup_epochs: Field


class DefaultsTrainingBaseData(BaseMixinData):
    main: DefaultsTrainingBaseGroupData
    fit: DefaultsTrainingBaseGroupData
    optimizer: DefaultsTrainingBaseGroupData
    outputs: DefaultsTrainingBaseGroupData
    checkpoint: DefaultsTrainingBaseGroupData


class ArchitectureMixinForm(BaseMixinData):
    def update(self, data: Any, prefix: str = "", **kwargs):
        for _key, _value in data.__fields__.items():
            name_ = f"{prefix}{_key}"
            if isinstance(_value.type_, ModelMetaclass):
                self.update(getattr(data, _key), f"{name_}_")
                continue
            _method_name = f"_set_{name_}"
            _method = getattr(self, _method_name, None)
            if _method:
                _method(getattr(data, _key), **kwargs)


class ArchitectureBaseForm(ArchitectureMixinForm):
    main: DefaultsTrainingBaseGroupData
    fit: DefaultsTrainingBaseGroupData
    optimizer: DefaultsTrainingBaseGroupData

    def _set_batch(self, value: int):
        fields = list(filter(lambda item: item.name == "batch", self.fit.fields))
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_type(self, value):
        fields = list(filter(lambda item: item.name == "optimizer", self.main.fields))
        if not fields:
            return
        fields[0].value = value


class ArchitectureBasicForm(ArchitectureBaseForm):
    outputs: DefaultsTrainingBaseGroupData
    checkpoint: DefaultsTrainingBaseGroupData

    def update(self, data: TrainData):
        print(data.native())

        for param in data.dict():
                if param == "batch":
                    self.fit.fields[0].value = data.batch
                elif param == "epochs":
                    self.fit.fields[1].value = data.epochs
                elif param == "optimizer":
                    self.fit.fields[2].value = data.optimizer.parameters.main.learning_rate
                    self.main.fields[0].value = data.optimizer.type.value

        

    def _set_architecture_parameters_outputs(self, value: List):
        for item in value:
            self.update(item, "architecture_parameters_outputs_", id=item.id)


class ArchitectureYoloForm(ArchitectureBaseForm):
    pass


class DefaultsTrainingData(BaseMixinData):
    base: DefaultsTrainingBaseData # ArchitectureBaseForm

    def update(self, dataset: DatasetData, training_base: TrainData):
        _class = getattr(
            sys.modules.get(__name__), f"Architecture{dataset.architecture}Form"
        )
        self.base = _class(
            **Architectures.get(dataset.architecture, ArchitectureChoice.Basic)
        )
        self.base.update(training_base)


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData

    def __update_training_outputs(self, layers: LayersList, training_data):
        outputs = {}
        for layer in layers:
            losses_data = {**TrainingLossSelect}
            training_task_rel = TrainingTasksRelations.get(layer.task)
            training_layer = training_data.base.architecture.parameters.outputs.get(
                layer.id
            )
            losses_list = list(
                map(
                    lambda item: {"label": item.value, "value": item.name},
                    training_task_rel.losses if training_task_rel else [],
                )
            )
            losses_data.update(
                {
                    "name": losses_data.get("name") % layer.id,
                    "parse": losses_data.get("parse") % layer.id,
                    "value": training_layer.loss
                    if training_layer
                    else (losses_list[0].get("label") if losses_list else ""),
                    "list": losses_list,
                }
            )
            metrics_data = {**TrainingMetricSelect}
            metrics_list = list(
                map(
                    lambda item: {"label": item.value, "value": item.name},
                    training_task_rel.metrics if training_task_rel else [],
                )
            )
            available_metrics = list(
                set(training_layer.metrics)
                & set(training_task_rel.metrics if training_task_rel else [])
            )
            metrics_data.update(
                {
                    "name": metrics_data.get("name") % layer.id,
                    "parse": metrics_data.get("parse") % layer.id,
                    "value": available_metrics
                    if available_metrics
                    else ([metrics_list[0].get("label")] if metrics_list else []),
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

    def update_by_model(self, model: ModelDetailsData, training_data):
        self.__update_training_outputs(model.outputs, training_data)
        self.__update_training_checkpoint(model.outputs)
