"""
## Структура данных параметров `output`-слоев
"""

from typing import List, Any
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import UniqueListMixin, IDMixinData
from ..presets.training import TasksGroups
from ..exceptions import ValueNotInListException
from .extra import TaskChoice, LossChoice, MetricChoice, TasksGroupsList
from . import callbacks


class OutputData(IDMixinData):
    """
    Информация о `output`-слое
    """

    classes_quantity: PositiveInt
    "Количество классов"
    task: TaskChoice
    "Задача"
    loss: LossChoice
    "Loss"
    metrics: List[MetricChoice]
    "Список метрик"
    callbacks: Any
    "Список колбэков"

    @validator("loss")
    def _validate_loss(cls, value: LossChoice, values) -> LossChoice:
        __task = values.get("task")
        __losses = TasksGroupsList(TasksGroups).get(__task).losses
        if value.value not in __losses:
            raise ValueNotInListException(value.value, __losses)
        return value

    @validator("metrics")
    def _validate_metrics(cls, value: MetricChoice, values) -> MetricChoice:
        __task = values.get("task")
        __metrics = TasksGroupsList(TasksGroups).get(__task).metrics
        __value = list(map(lambda item: item.value, value))
        __not_in_list = list(set(__value) - set(__metrics))
        if __not_in_list:
            raise ValueNotInListException(__not_in_list, __metrics)
        return value

    @validator("task", pre=True)
    def _validate_task(cls, value: TaskChoice) -> TaskChoice:
        if value not in list(TaskChoice):
            raise EnumMemberError(enum_values=list(TaskChoice))
        name = (value if isinstance(value, TaskChoice) else TaskChoice(value)).name
        type_ = getattr(callbacks, getattr(callbacks.Callback, name))
        cls.__fields__["callbacks"].type_ = type_
        return value

    @validator("callbacks", always=True)
    def _validate_callbacks(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class OutputsList(UniqueListMixin):
    """
    Список `outputs`-слоев, основанных на `OutputData`
    """

    class Meta:
        source = OutputData
        identifier = "id"
