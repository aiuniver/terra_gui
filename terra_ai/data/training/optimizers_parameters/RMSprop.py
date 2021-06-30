"""
## Оптимайзер `RMSprop`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGt0, ConstrainedFloatValueGe0


class ParametersExtraData(BaseMixinData):
    rho: ConstrainedFloatValueGt0 = 0.9
    momentum: ConstrainedFloatValueGe0 = 0.0
    epsilon: ConstrainedFloatValueGt0 = 1e-07
    centered: bool = False
