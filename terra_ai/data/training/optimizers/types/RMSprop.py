"""
## Оптимайзер `RMSprop`
"""

from pydantic.types import PositiveFloat

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0


class ParametersExtraData(BaseMixinData):
    rho: PositiveFloat = 0.9
    momentum: ConstrainedFloatValueGe0 = 0.0
    epsilon: PositiveFloat = 1e-07
    centered: bool = False
