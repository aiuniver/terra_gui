from terra_ai.data.datasets.creations.layers.extra import MinMaxScalerData
from terra_ai.data.datasets.extra import LayerScalerRegressionChoice
from terra_ai.data.mixins import BaseMixinData
from pydantic.types import PositiveInt
from typing import Optional


class ParametersData(BaseMixinData, MinMaxScalerData):
    """
    Обработчик типа задачи "регрессия".
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    scaler: LayerScalerRegressionChoice
    # Внутренние параметры
    put: Optional[PositiveInt]
