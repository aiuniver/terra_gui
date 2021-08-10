"""
## Структура данных чекпоинтов
"""

from ..mixins import BaseMixinData
from ..types import IDType
from .extra import CheckpointIndicatorChoice, CheckpointModeChoice, CheckpointTypeChoice


class CheckpointData(BaseMixinData):
    layer: IDType
    type: CheckpointTypeChoice = CheckpointTypeChoice.Metrics
    indicator: CheckpointIndicatorChoice = CheckpointIndicatorChoice.Val
    mode: CheckpointModeChoice = CheckpointModeChoice.Max
    save_best: bool = True
    save_weights: bool = False
