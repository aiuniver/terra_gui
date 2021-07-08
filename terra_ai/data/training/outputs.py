"""
## Структура данных параметров `output`-слоев
"""

from typing import List, Optional, Any, Union
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import UniqueListMixin, AliasMixinData
from ..presets.training import TasksGroups
from ..exceptions import ValueNotInListException
from .extra import TaskChoice, LossChoice, MetricChoice
from . import callbacks


class OutputData(AliasMixinData):
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
    callbacks: Optional[Any]
    "Список колбэков"

    @validator("loss", allow_reuse=True)
    def _validate_loss(cls, value: LossChoice, values) -> LossChoice:
        __task = values.get("task")
        __losses = TasksGroups.get(__task).losses
        if value.value not in __losses:
            raise ValueNotInListException(value.value, __losses)
        return value

    @validator("metrics", allow_reuse=True)
    def _validate_metrics(cls, value: MetricChoice, values) -> MetricChoice:
        __task = values.get("task")
        __metrics = TasksGroups.get(__task).metrics
        __value = list(map(lambda item: item.value, value))
        if list(set(__value) - set(__metrics)):
            raise ValueNotInListException(__value, __metrics)
        return value

    @validator("task", allow_reuse=True, pre=True)
    def _validate_task(cls, value: TaskChoice) -> TaskChoice:
        if not hasattr(TaskChoice, value):
            raise EnumMemberError(enum_values=list(TaskChoice))
        type_ = getattr(callbacks, getattr(callbacks.TaskCallbacksType, value))
        cls.__fields__["callbacks"].type_ = type_
        cls.__fields__["callbacks"].required = True
        return value

    @validator("callbacks", allow_reuse=True)
    def _validate_callbacks(
        cls, value: Any, **kwargs
    ) -> Union[callbacks.TaskCallbacksTypeUnion]:
        return kwargs.get("field").type_(**value)


class OutputsList(UniqueListMixin):
    """
    Список `outputs`-слоев, основанных на `OutputData`
    """

    class Meta:
        source = OutputData
        identifier = "alias"
