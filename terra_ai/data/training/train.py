"""
## Структура данных обучения
"""

from pydantic.types import PositiveInt

from ..mixins import BaseMixinData
from .optimizer import OptimizerData
from .outputs import OutputsList


class TrainData(BaseMixinData):
    """
    Параметры обучения
    """

    batch: PositiveInt = 32
    epochs: PositiveInt = 20
    optimizer: OptimizerData
    outputs: OutputsList
