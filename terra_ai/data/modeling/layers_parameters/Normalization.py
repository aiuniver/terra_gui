"""
## Тип слоя `Normalization`
"""

from typing import Optional
from pydantic.types import PositiveFloat

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0Le1
from .extra import InitializerChoice, RegularizerChoice, ConstraintChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
    mean: Optional[float] = None
    variance: Optional[float] = None
