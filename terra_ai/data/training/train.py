"""
## Структура данных обучения
"""

from pydantic import validator
from pydantic.types import PositiveInt

from ..mixins import BaseMixinData
from ..exceptions import ValueNotInListException
from .optimizer import OptimizerData
from .outputs import OutputsList
from .checkpoint import CheckpointData


class TrainData(BaseMixinData):
    """
    Параметры обучения
    """

    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData
    outputs: OutputsList
    checkpoint: CheckpointData

    @validator("checkpoint", allow_reuse=True)
    def _validate_checkpoint_layer(
        cls, value: CheckpointData, values
    ) -> CheckpointData:
        __layers = values.get("outputs").ids
        if value.layer not in __layers:
            raise ValueNotInListException(value.layer, __layers)
        return value
