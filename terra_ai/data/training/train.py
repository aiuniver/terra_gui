"""
## Структура данных обучения
"""

import json
from typing import Any, Optional, List
from pydantic import validator
from pydantic.types import conint, confloat, PositiveInt
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
    box_channel: conint(ge=0, le=2) = 1
    num_examples: conint(ge=1, le=10) = 10
    sensitivity: confloat(gt=0, le=1) = 0.15
    threashold: confloat(gt=0, le=1) = 0.1
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
    box_channel: conint(ge=0, le=2) = 1
    autoupdate: bool = False
    sensitivity: confloat(gt=0, le=1) = 0.15
    threashold: confloat(gt=0, le=1) = 0.1


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
        return field.type_(**(value or {}))


class ArchitectureData(BaseMixinData):
    model: Any
    type: ArchitectureChoice
    parameters: Any

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"model"}})
        return super().dict(**kwargs)

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
        if not value:
            return value
        _model = values.get("model")
        _outputs = value.get("outputs", [])
        for _index, _output in enumerate(_outputs):
            _output["task"] = str(_model.layers.get(_output.get("id")).task)
            _outputs[_index] = _output
        value["outputs"] = _outputs
        return field.type_(**(value or {}))


class TrainData(BaseMixinData):
    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData = OptimizerData(type=OptimizerChoice.Adam)
    architecture: ArchitectureData = ArchitectureData(type=ArchitectureChoice.Basic)
