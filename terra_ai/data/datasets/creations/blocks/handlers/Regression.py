from pydantic.types import PositiveInt
from typing import Optional

from terra_ai.data.datasets.extra import LayerScalerRegressionChoice
from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    """
    Обработчик типа задачи "регрессия".
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    scaler: LayerScalerRegressionChoice
