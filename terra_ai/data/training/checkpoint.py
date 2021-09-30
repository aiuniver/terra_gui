"""
## Структура данных чекпоинтов
"""

from pydantic.types import PositiveInt

from ..mixins import BaseMixinData
from .extra import (
    CheckpointIndicatorChoice,
    CheckpointModeChoice,
    CheckpointTypeChoice,
    MetricChoice,
)


class CheckpointData(BaseMixinData):
    layer: PositiveInt
    type: CheckpointTypeChoice = CheckpointTypeChoice.Metrics
    metric_name: MetricChoice = MetricChoice.Accuracy
    indicator: CheckpointIndicatorChoice = CheckpointIndicatorChoice.Val
    mode: CheckpointModeChoice = CheckpointModeChoice.Max
    save_best: bool = True
    save_weights: bool = False
