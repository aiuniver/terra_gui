from terra_ai.data.datasets.extra import LayerScalerTimeseriesChoice
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilepath
from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt


class ParametersData(BaseMixinData):
    # sources_paths: List[confilepath(ext="csv")]
    length: PositiveInt
    step: PositiveInt
    trend: bool
    trend_limit: Optional[str]
    depth: Optional[PositiveInt]
    scaler: Optional[LayerScalerTimeseriesChoice]
    # Внутренние параметры
    put: Optional[PositiveInt]
    # transpose: Optional[bool]
    # separator: Optional[str]

    @validator("trend")
    def _validate_trend(cls, value: bool) -> bool:
        if value:
            cls.__fields__["trend_limit"].required = True
        else:
            cls.__fields__["depth"].required = True
            cls.__fields__["scaler"].required = True
        return value
