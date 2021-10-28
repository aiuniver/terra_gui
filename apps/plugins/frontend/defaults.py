import sys

from typing import List, Dict, Optional, Union, Any
from pydantic import validator
from pydantic.main import ModelMetaclass

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.layer import LayersList
from terra_ai.data.training.extra import ArchitectureChoice, TasksRelations

from .presets.defaults.training import (
    TrainingLossSelect,
    TrainingMetricSelect,
    TrainingClassesQuantitySelect,
    ArchitectureOptimizerExtraFields,
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

    def set_layer_datatype(self, dataset: DatasetData = None):
        _inputs = []
        _outputs = []
        if dataset:
            for _index, _item in dataset.inputs.items():
                _inputs.append(
                    {"value": _index, "label": f"{_item.name} [shape={_item.shape}]"}
                )
            for _index, _item in dataset.outputs.items():
                _outputs.append(
                    {"value": _index, "label": f"{_item.name} [shape={_item.shape}]"}
                )
        self.layer_form[2].list = _inputs
        self.layer_form[3].list = _outputs


class DefaultsTrainingBaseGroupData(BaseMixinData):
    name: Optional[str]
    collapsable: bool = False
    collapsed: bool = False
    visible: bool = True
    fields: Union[List[Field], Dict[str, List[Field]]]


class ArchitectureMixinForm(BaseMixinData):
    def update(self, data: Any, prefix: str = "", **kwargs):
        if not data:
            return
        for _key, _value in data.__fields__.items():
            name_ = f"{prefix}{_key}"
            if isinstance(_value.type_, ModelMetaclass):
                self.update(getattr(data, _key), f"{name_}_")
                continue
            _method_name = f"_set_{name_}"
            _method = getattr(self, _method_name, None)
            if _method:
                _method(getattr(data, _key), **kwargs)


class ArchitectureBaseGroupForm(ArchitectureMixinForm):
    main: DefaultsTrainingBaseGroupData
    fit: DefaultsTrainingBaseGroupData
    optimizer: DefaultsTrainingBaseGroupData

    def _set_optimizer_type(self, value, **kwargs):
        fields = list(filter(lambda item: item.name == "optimizer", self.main.fields))
        if not fields:
            return
        self.optimizer.fields = []
        for item in ArchitectureOptimizerExtraFields.get(value):
            self.optimizer.fields.append(Field(**item))
        fields[0].value = value

    def _set_batch(self, value: int, **kwargs):
        fields = list(filter(lambda item: item.name == "batch", self.fit.fields))
        if not fields:
            return
        fields[0].value = value

    def _set_epochs(self, value: int, **kwargs):
        fields = list(filter(lambda item: item.name == "epochs", self.fit.fields))
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_main_learning_rate(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_main_learning_rate",
                self.fit.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_beta_1(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_beta_1",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_beta_2(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_beta_2",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_epsilon(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_epsilon",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_amsgrad(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_amsgrad",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_nesterov(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_nesterov",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_momentum(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_momentum",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_centered(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_centered",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_rho(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_rho", self.optimizer.fields
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_optimizer_parameters_extra_initial_accumulator_value(
        self, value, **kwargs
    ):
        fields = list(
            filter(
                lambda item: item.name == "optimizer_extra_initial_accumulator_value",
                self.optimizer.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value


class ArchitectureOutputsGroupFrom(ArchitectureMixinForm):
    outputs: DefaultsTrainingBaseGroupData

    def update(self, data: Any, prefix: str = "", **kwargs):
        model = kwargs.get("model")
        if model and model.outputs:
            self._update_outputs(model.outputs, data)

        return super().update(data, prefix=prefix, **kwargs)

    def _update_outputs(self, layers: LayersList, data):
        outputs = {}
        for layer in layers:
            _task_rel = TasksRelations.get(layer.task)
            _layer_data = data.architecture.parameters.outputs.get(layer.id)

            _losses_rel = _task_rel.losses if _task_rel else []
            _losses_list = list(
                map(
                    lambda item: {"label": item.value, "value": item.name},
                    _losses_rel,
                )
            )
            _losses_value = None
            if _layer_data and _losses_rel:
                _losses_value = (
                    _layer_data.loss
                    if _layer_data.loss in _losses_rel
                    else _losses_rel[0]
                )
            _losses_data = {**TrainingLossSelect}
            _losses_data.update(
                {
                    "name": _losses_data.get("name") % layer.id,
                    "parse": _losses_data.get("parse") % layer.id,
                    "value": _losses_value,
                    "list": _losses_list,
                }
            )

            _metrics_rel = _task_rel.metrics if _task_rel else []
            _metrics_list = list(
                map(
                    lambda item: {"label": item.value, "value": item.name},
                    _metrics_rel,
                )
            )
            _metrics_value = []
            if _layer_data and _metrics_rel:
                _metrics_value = list(set(_layer_data.metrics) & set(_metrics_rel))
            _metrics_data = {**TrainingMetricSelect}
            _metrics_data.update(
                {
                    "name": _metrics_data.get("name") % layer.id,
                    "parse": _metrics_data.get("parse") % layer.id,
                    "value": _metrics_value,
                    "list": _metrics_list,
                    "changeable": True,
                }
            )

            _classes_quantity_data = {**TrainingClassesQuantitySelect}
            _classes_quantity_data.update(
                {
                    "name": _classes_quantity_data.get("name") % layer.id,
                    "parse": _classes_quantity_data.get("parse") % layer.id,
                    "value": layer.num_classes,
                }
            )

            outputs.update(
                {
                    layer.id: {
                        "name": f"Слой «{layer.name}»",
                        "classes_quantity": layer.num_classes,
                        "fields": [
                            Field(**_losses_data),
                            Field(**_metrics_data),
                            Field(**_classes_quantity_data),
                        ],
                    }
                }
            )
        self.outputs.fields = outputs


class ArchitectureBaseForm(ArchitectureOutputsGroupFrom, ArchitectureBaseGroupForm):
    pass


class ArchitectureBasicForm(ArchitectureBaseForm):
    checkpoint: DefaultsTrainingBaseGroupData

    def _set_architecture_parameters_checkpoint_metric_name(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name
                == "architecture_parameters_checkpoint_metric_name",
                self.checkpoint.fields,
            )
        )
        if not fields:
            return
        fields_layer = list(
            filter(
                lambda item: item.name == "architecture_parameters_checkpoint_layer",
                self.checkpoint.fields,
            )
        )
        if not fields_layer:
            return
        fields_outputs = list(
            filter(
                lambda item: item.name
                == f"architecture_parameters_outputs_{fields_layer[0].value}_metrics",
                self.outputs.fields.get(fields_layer[0].value, {}).get("fields"),
            )
        )
        if not fields_outputs:
            return
        fields[0].list = list(
            map(
                lambda item: {"value": item.name, "label": item.value},
                fields_outputs[0].value,
            )
        )
        fields[0].value = (
            value if value in fields_outputs[0] else fields_outputs[0].value[0]
        )

    def _set_architecture_parameters_checkpoint_layer(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "architecture_parameters_checkpoint_layer",
                self.checkpoint.fields,
            )
        )
        if not fields:
            return
        fields[0].list = list(
            map(
                lambda item: {"value": item[0], "label": item[1].get("name")},
                self.outputs.fields.items(),
            )
        )
        fields[0].value = (
            value
            if value in self.outputs.fields.keys()
            else list(self.outputs.fields.keys())[0]
        )

    def _set_architecture_parameters_checkpoint_type(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "architecture_parameters_checkpoint_type",
                self.checkpoint.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_architecture_parameters_checkpoint_indicator(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name
                == "architecture_parameters_checkpoint_indicator",
                self.checkpoint.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_architecture_parameters_checkpoint_mode(self, value, **kwargs):
        fields = list(
            filter(
                lambda item: item.name == "architecture_parameters_checkpoint_mode",
                self.checkpoint.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value


class ArchitectureYoloBaseForm(ArchitectureBaseForm):
    yolo: DefaultsTrainingBaseGroupData

    def _set_architecture_parameters_yolo_train_lr_init(self, value):
        fields = list(
            filter(
                lambda item: item.name == "train_lr_init",
                self.yolo.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_architecture_parameters_yolo_train_lr_end(self, value):
        fields = list(
            filter(
                lambda item: item.name == "train_lr_end",
                self.yolo.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_architecture_parameters_yolo_yolo_iou_loss_thresh(self, value):
        fields = list(
            filter(
                lambda item: item.name == "yolo_iou_loss_thresh",
                self.yolo.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value

    def _set_architecture_parameters_yolo_train_warmup_epochs(self, value):
        fields = list(
            filter(
                lambda item: item.name == "train_warmup_epochs",
                self.yolo.fields,
            )
        )
        if not fields:
            return
        fields[0].value = value


class ArchitectureYoloV3Form(ArchitectureYoloBaseForm):
    pass


class ArchitectureYoloV4Form(ArchitectureYoloBaseForm):
    pass


class DefaultsTrainingData(BaseMixinData):
    architecture: ArchitectureChoice
    base: Optional[ArchitectureBaseForm]

    def __init__(self, project: Any = None, **data):
        data.update({"base": Architectures.get(data.get("architecture", "Base"))})
        super().__init__(**data)
        if project:
            self._update(project)

    @validator("architecture", pre=True)
    def _validate_architecture(cls, value: ArchitectureChoice) -> ArchitectureChoice:
        cls.__fields__["base"].required = True
        cls.__fields__["base"].type_ = getattr(
            sys.modules.get(__name__), f"Architecture{value}Form"
        )
        return value

    @validator("base", pre=True)
    def _validate_base(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})

    def _update(self, project: Any):
        self.base.update(project.training.base, model=project.model)


class DefaultsData(BaseMixinData):
    datasets: DefaultsDatasetsData
    modeling: DefaultsModelingData
    training: DefaultsTrainingData
