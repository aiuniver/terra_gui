from pydantic import PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    length: PositiveInt
    step: PositiveInt
    trend_limit: str

    # Внутренние параметры
    depth: PositiveInt = 1
