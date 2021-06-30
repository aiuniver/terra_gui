"""
## Тип слоя `ReLU`
"""

from typing import Optional
from pydantic.types import confloat

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    max_value: Optional[confloat(ge=0)]
    negative_slope: confloat(ge=0) = 0
    threshold: confloat(ge=0) = 0
