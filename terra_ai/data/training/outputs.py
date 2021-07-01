"""
## Структура данных параметров `output`-слоев
"""

import sys

from enum import Enum
from typing import List, Optional, Any, Union
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import UniqueListMixin, AliasMixinData, BaseMixinData
from ..presets.training import TasksGroups
from ..exceptions import TaskGroupException
from .extra import TaskChoice, LossChoice, MetricChoice


class TaskCallbacksTypeclassificationData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_loss_for_classes: bool = True
    plot_metric_for_classes: bool = True
    show_best_images: bool = True
    show_worst_images: bool = False
    plot_final: bool = True


class TaskCallbacksTypesegmentationData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_loss_for_classes: bool = True
    plot_metric_for_classes: bool = True
    show_best_images: bool = True
    show_worst_images: bool = False
    plot_final: bool = True


class TaskCallbacksTyperegressionData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_scatter: bool = True
    plot_final: bool = True


class TaskCallbacksTypetimeseriesData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_autocorrelation: bool = True
    plot_pred_and_true: bool = True
    plot_final: bool = True


TaskCallbacksType = Enum(
    "TaskCallbacksType",
    dict(
        map(
            lambda item: (item, f"TaskCallbacksType{item}Data"),
            list(TaskChoice),
        )
    ),
    type=str,
)
"""
Список возможных колбэков в таске
"""


TaskCallbacksTypeUnion = tuple(
    map(
        lambda item: getattr(sys.modules.get(__name__), item),
        TaskCallbacksType,
    )
)
"""
Список возможных колбэков в таске в виде классов
"""


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
            raise TaskGroupException(value.value, __losses)
        return value

    @validator("metrics", allow_reuse=True)
    def _validate_metrics(cls, value: MetricChoice, values) -> MetricChoice:
        __task = values.get("task")
        __metrics = TasksGroups.get(__task).metrics
        __value = list(map(lambda item: item.value, value))
        if list(set(__value) - set(__metrics)):
            raise TaskGroupException(__value, __metrics)
        return value

    @validator("task", allow_reuse=True, pre=True)
    def _validate_task(cls, value: TaskChoice) -> TaskChoice:
        if not hasattr(TaskChoice, value):
            raise EnumMemberError(enum_values=list(TaskChoice))
        type_ = getattr(sys.modules.get(__name__), getattr(TaskCallbacksType, value))
        cls.__fields__["callbacks"].type_ = type_
        cls.__fields__["callbacks"].required = True
        return value

    @validator("callbacks", allow_reuse=True)
    def _validate_callbacks(cls, value: Any, **kwargs) -> Union[TaskCallbacksTypeUnion]:
        return kwargs.get("field").type_(**value)


class OutputsList(UniqueListMixin):
    """
    Список `outputs`-слоев, основанных на `OutputData`
    """

    class Meta:
        source = OutputData
        identifier = "alias"
