"""
## Структура данных чекпоинтов
"""

from ..mixins import BaseMixinData
from ..types import AliasType
from .extra import CheckpointIndicatorChoice, CheckpointModeChoice, CheckpointTypeChoice


class CheckpointData(BaseMixinData):
    layer: AliasType
    type: CheckpointTypeChoice = CheckpointTypeChoice.metrics
    indicator: CheckpointIndicatorChoice = CheckpointIndicatorChoice.val
    mode: CheckpointModeChoice = CheckpointModeChoice.max
    save_best: bool = True
    save_weights: bool = False
