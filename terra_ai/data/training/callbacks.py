"""
## Структура данных колбэков
"""

import sys

from enum import Enum

from ..mixins import BaseMixinData
from .extra import TaskChoice


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
