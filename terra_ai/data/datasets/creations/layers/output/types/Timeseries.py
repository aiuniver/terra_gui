from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt

from ...extra import MinMaxScalerData
from .....extra import LayerScalerTimeseriesChoice
from ......types import confilepath


class ParametersData(MinMaxScalerData):
    sources_paths: List[confilepath(ext="csv")]
    length: PositiveInt
    step: PositiveInt
    trend: bool
    trend_limit: Optional[str]
    depth: Optional[PositiveInt]
    scaler: Optional[LayerScalerTimeseriesChoice]

    cols_names: Optional[List[str]]

    @validator("trend")
    def _validate_trend(cls, value: bool) -> bool:
        if value:
            cls.__fields__["trend_limit"].required = True
        else:
            cls.__fields__["depth"].required = True
            cls.__fields__["scaler"].required = True
        return value
