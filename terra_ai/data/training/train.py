"""
## Структура данных обучения
"""

from ..mixins import BaseMixinData
from .optimizer import OptimizerData


class TrainData(BaseMixinData):
    """
    Параметры обучения
    """

    optimizer: OptimizerData
