"""
## Оптимайзер `Adadelta`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGt0


class ParametersExtraData(BaseMixinData):
    rho: ConstrainedFloatValueGt0 = 0.95
    epsilon: ConstrainedFloatValueGt0 = 1e-07
