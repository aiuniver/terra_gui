from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    length: PositiveInt
    step: PositiveInt
    # trend: bool
    # trend_limit: Optional[str]
    depth: PositiveInt

    # @validator("trend")
    # def _validate_trend(cls, value: bool) -> bool:
    #     if value:
    #         cls.__fields__["trend_limit"].required = True
    #     else:
    #         cls.__fields__["depth"].required = True
    #         cls.__fields__["scaler"].required = True
    #     return value
