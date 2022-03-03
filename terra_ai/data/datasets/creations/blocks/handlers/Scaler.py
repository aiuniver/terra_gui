from typing import Optional
from pydantic.types import PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    """
    Обработчик типа задачи "скейлер".
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    # Внутренние параметры
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    depth: Optional[PositiveInt]
