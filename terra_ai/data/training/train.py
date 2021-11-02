"""
## Структура данных обучения
"""

import os
import json
import shutil

from pathlib import Path
from typing import Any, Optional, List
from pydantic import validator, PrivateAttr
from pydantic.types import conint, confloat, PositiveInt
from pydantic.errors import EnumMemberError

from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.mixins import BaseMixinData, UniqueListMixin, IDMixinData
from terra_ai.data.training import optimizers, architectures
from terra_ai.data.training.extra import (
    OptimizerChoice,
    ArchitectureChoice,
    LossGraphShowChoice,
    MetricGraphShowChoice,
    ExampleChoiceTypeChoice,
    BalanceSortedChoice,
    MetricChoice,
    StateStatusChoice,
)


DEFAULT_TRAINING_PATH_NAME = "__current"
CONFIG_TRAINING_FILENAME = "config.json"


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
    train: StateButtonData
    stop: StateButtonData
    clear: StateButtonData
    save: StateButtonData


class StateData(BaseMixinData):
    status: StateStatusChoice
    buttons: Optional[StateButtonsData]

    @staticmethod
    def get_buttons(status: StateStatusChoice) -> dict:
        value = {
            "train": {"title": "Обучить", "visible": True},
            "stop": {"title": "Остановить", "visible": False},
            "clear": {"title": "Сбросить", "visible": False},
            "save": {"title": "Сохранить", "visible": False},
        }
        if status in [StateStatusChoice.training, StateStatusChoice.addtrain]:
            value["train"].update({"visible": False, "title": "Возобновить"})
            value["stop"].update({"visible": True})
            value["clear"].update({"visible": False})
            value["save"].update({"visible": False})
        elif status == StateStatusChoice.trained:
            value["train"].update({"visible": True, "title": "Дообучить"})
            value["stop"].update({"visible": False})
            value["clear"].update({"visible": True})
            value["save"].update({"visible": True})
        elif status == StateStatusChoice.stopped:
            value["train"].update({"visible": True, "title": "Возобновить"})
            value["stop"].update({"visible": False})
            value["clear"].update({"visible": True})
            value["save"].update({"visible": True})
        return StateButtonsData(**value)

    @validator("buttons", always=True)
    def _validate_buttons(cls, value: StateButtonsData, values) -> StateButtonsData:
        return StateData.get_buttons(values.get("status"))

    def set(self, value: str):
        self.status = StateStatusChoice[value]
        self.buttons = StateData.get_buttons(self.status)


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
            _output["task"] = (
                _model.layers.get(_output.get("id")).task.value
                if _model.layers.get(_output.get("id")).task
                else None
            )
            _outputs[_index] = _output
        value["outputs"] = _outputs
        value["model"] = _model
        return field.type_(**(value or {}))


class TrainData(BaseMixinData):
    model: Any
    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData = OptimizerData(type=OptimizerChoice.Adam)
    architecture: ArchitectureData = ArchitectureData(type=ArchitectureChoice.Basic)

    @validator("architecture", pre=True, allow_reuse=True)
    def _validate_architecture(cls, value, values):
        if not value:
            value = {}
        value.update({"model": values.get("model")})
        return value

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"model"}})
        return super().dict(**kwargs)


class TrainingDetailsData(BaseMixinData):
    model: Any
    name: Optional[str]
    base: TrainData = TrainData()
    interactive: InteractiveData = InteractiveData()
    state: StateData = StateData(status="no_train")
    result: Optional[dict]
    deploy: Optional[DeployData]

    _path: Path = PrivateAttr()

    def __init__(self, **data):
        self._path = Path(data.get("path"))

        _name = data.get("name", DEFAULT_TRAINING_PATH_NAME)
        if _name == DEFAULT_TRAINING_PATH_NAME:
            os.makedirs(Path(self._path, DEFAULT_TRAINING_PATH_NAME), exist_ok=True)

        if not Path(self._path, _name).is_dir():
            _name = DEFAULT_TRAINING_PATH_NAME

        config = Path(self._path, _name, CONFIG_TRAINING_FILENAME)
        if not config.is_file():
            with open(config, "w") as config_ref:
                json.dump({"name": _name}, config_ref)
        with open(config) as config_ref:
            data = json.load(config_ref)
        data["name"] = _name

        super().__init__(**data)

    @property
    def path(self) -> Path:
        return Path(self._path, self.name)

    @validator("base", pre=True, allow_reuse=True)
    def _validate_base(cls, value, values):
        if not value:
            value = {}
        value.update({"model": values.get("model")})
        return value

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"model"}})
        return super().dict(**kwargs)

    def rename(self, value: str):
        source = self.path
        self.name = value
        shutil.rmtree(self.path, ignore_errors=True)
        shutil.move(source, self.path)

    def clear(self):
        print(self.path)
