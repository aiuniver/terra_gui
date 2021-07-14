"""
## Параметры колбэков
"""

from enum import Enum

from ...mixins import BaseMixinData
from ..extra import TaskChoice

from . import types


class CallbackBaseData(BaseMixinData):
    pass


class CallbackClassificationData(CallbackBaseData, types.Classification.ParametersData):
    pass


class CallbackSegmentationData(CallbackBaseData, types.Segmentation.ParametersData):
    pass


class CallbackRegressionData(CallbackBaseData, types.Regression.ParametersData):
    pass


class CallbackTimeseriesData(CallbackBaseData, types.Timeseries.ParametersData):
    pass


Callback = Enum(
    "Callback",
    dict(map(lambda item: (item.name, f"Callback{item.name}Data"), list(TaskChoice))),
    type=str,
)
