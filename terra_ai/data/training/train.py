"""
## Структура данных обучения
"""

import json
from typing import Any, List, Optional
from pydantic import validator
from pydantic.types import conint, PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData, UniqueListMixin, IDMixinData
from . import optimizers
from . import architectures
from .extra import (
    OptimizerChoice,
    ArchitectureChoice,
    LossGraphShowChoice,
    MetricGraphShowChoice,
    ExampleChoiceTypeChoice,
    BalanceSortedChoice,
    MetricChoice,
    StateStatusChoice,
)
from ..types import ConstrainedFloatValueGe0Le1
from terra_ai.utils import decamelize
from terra_ai.data.modeling.model import ModelDetailsData
from apps.plugins.frontend.presets.defaults.training import TrainingTasksRelations
from terra_ai.data.training.checkpoint import CheckpointData
from terra_ai.data.training.outputs import OutputsList


class LossGraphData(IDMixinData):
    output_idx: PositiveInt
    show: LossGraphShowChoice


class LossGraphsList(UniqueListMixin):
    class Meta:
        source = LossGraphData
        identifier = "id"


class MetricGraphData(IDMixinData):
    output_idx: PositiveInt
    show: MetricGraphShowChoice
    show_metric: Optional[MetricChoice]


class MetricGraphsList(UniqueListMixin):
    class Meta:
        source = MetricGraphData
        identifier = "id"


class IntermediateResultData(BaseMixinData):
    show_results: bool = False
    example_choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.seed
    main_output: Optional[PositiveInt]
    num_examples: conint(ge=1, le=10) = 10
    show_statistic: bool = False
    autoupdate: bool = False


class YoloIntermediateResultData(BaseMixinData):
    show_results: bool = False
    example_choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.seed
    box_channel: conint(ge=0, le=2) = 1
    num_examples: conint(ge=1, le=10) = 10
    sensitivity: ConstrainedFloatValueGe0Le1 = 0.15
    threashold: ConstrainedFloatValueGe0Le1 = 0.1
    show_statistic: bool = False
    autoupdate: bool = False


class ProgressTableData(BaseMixinData):
    output_idx: PositiveInt
    show_loss: bool = True
    show_metrics: bool = True


class ProgressTableList(UniqueListMixin):
    class Meta:
        source = ProgressTableData
        identifier = "output_idx"


class StatisticData(BaseMixinData):
    output_id: List[PositiveInt] = []
    autoupdate: bool = False


class YoloStatisticData(BaseMixinData):
    box_channel: conint(ge=0, le=2) = 1
    autoupdate: bool = False
    sensitivity: ConstrainedFloatValueGe0Le1 = 0.15
    threashold: ConstrainedFloatValueGe0Le1 = 0.1


class BalanceData(BaseMixinData):
    show_train: bool = True
    show_val: bool = True
    sorted: BalanceSortedChoice = BalanceSortedChoice.descending


class InteractiveData(BaseMixinData):
    loss_graphs: LossGraphsList = LossGraphsList()
    metric_graphs: MetricGraphsList = MetricGraphsList()
    intermediate_result: IntermediateResultData = IntermediateResultData()
    progress_table: ProgressTableList = ProgressTableList()
    statistic_data: StatisticData = StatisticData()
    data_balance: BalanceData = BalanceData()


class YoloInteractiveData(BaseMixinData):
    loss_graphs: LossGraphsList = LossGraphsList()
    metric_graphs: MetricGraphsList = MetricGraphsList()
    intermediate_result: YoloIntermediateResultData = YoloIntermediateResultData()
    progress_table: ProgressTableList = ProgressTableList()
    statistic_data: YoloStatisticData = YoloStatisticData()
    data_balance: BalanceData = BalanceData()


class StateButtonData(BaseMixinData):
    title: str
    visible: bool


class StateButtonsData(BaseMixinData):
    train: StateButtonData = StateButtonData(title="Обучить", visible=False)
    stop: StateButtonData = StateButtonData(title="Остановить", visible=False)
    clear: StateButtonData = StateButtonData(title="Сбросить", visible=False)
    save: StateButtonData = StateButtonData(title="Сохранить", visible=False)


class StateData(BaseMixinData):
    status: StateStatusChoice = StateStatusChoice.no_train
    buttons: StateButtonsData = StateButtonsData()


class OptimizerData(BaseMixinData):
    type: OptimizerChoice
    parameters: Any

    @property
    def parameters_dict(self) -> dict:
        __data = json.loads(self.parameters.main.json())
        __data.update(json.loads(self.parameters.extra.json()))
        return __data

    @validator("type", pre=True)
    def _validate_type(cls, value: OptimizerChoice) -> OptimizerChoice:
        if value not in list(OptimizerChoice):
            raise EnumMemberError(enum_values=list(OptimizerChoice))
        name = (
            value if isinstance(value, OptimizerChoice) else OptimizerChoice(value)
        ).name
        type_ = getattr(optimizers, getattr(optimizers.Optimizer, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class ArchitectureData(BaseMixinData):
    type: ArchitectureChoice
    parameters: Any

    @property
    def outputs_dict(self) -> dict:
        __data = json.loads(self.parameters.outputs.json())
        return __data

    @validator("type", pre=True)
    def _validate_type(cls, value: ArchitectureChoice) -> ArchitectureChoice:
        if value not in list(ArchitectureChoice):
            raise EnumMemberError(enum_values=list(ArchitectureChoice))
        name = (
            value
            if isinstance(value, ArchitectureChoice)
            else ArchitectureChoice(value)
        ).name
        type_ = getattr(architectures, getattr(architectures.Architecture, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class TrainData(BaseMixinData):
    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData = OptimizerData(type=OptimizerChoice.Adam)
    architecture: ArchitectureData = ArchitectureData(type=ArchitectureChoice.Basic)

    def _update_arch_basic(self, model: ModelDetailsData):
        outputs = []
        for layer in model.outputs:
            training_layer = self.architecture.parameters.outputs.get(
                layer.id
            )
            training_task_rel = TrainingTasksRelations.get(layer.task)
            training_losses = (
                list(map(lambda item: item.name, training_task_rel.losses))
                if training_task_rel
                else None
            )
            training_metrics = (
                list(map(lambda item: item.name, training_task_rel.metrics))
                if training_task_rel
                else None
            )
            need_loss = training_layer.loss if training_layer else None
            if need_loss:
                loss = need_loss if need_loss in training_losses else training_losses[0]
            else:
                loss = training_losses[0] if training_losses else None
            need_metrics = training_layer.metrics if training_layer else []
            metrics = list(set(need_metrics) & set(training_metrics or []))
            outputs.append(
                {
                    "id": layer.id,
                    "classes_quantity": layer.num_classes,
                    "task": layer.task,
                    "loss": loss,
                    "metrics": metrics
                    if len(metrics)
                    else ([training_metrics[0]] if training_metrics else []),
                }
            )
        self.architecture.parameters.outputs = OutputsList(outputs)
        if model.outputs:
            checkpoint_data = {"layer": self.architecture.parameters.outputs[0].id}
            if self.architecture.parameters.checkpoint:
                checkpoint_data = (
                    self.architecture.parameters.checkpoint.native()
                )
                if not checkpoint_data.get("layer"):
                    checkpoint_data.update({"layer": self.model.outputs[0].id})
            self.architecture.parameters.checkpoint = CheckpointData(
                **checkpoint_data
            )

    def update_by_model(self, model: ModelDetailsData):
        _method_name = f"_update_arch_{decamelize(self.architecture.type)}"
        _method = getattr(self, _method_name, None)

        if _method:
            _method(model)
