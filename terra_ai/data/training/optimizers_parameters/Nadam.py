"""
## Оптимайзер `Nadam`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGt0


class ParametersExtraData(BaseMixinData):
    beta_1: ConstrainedFloatValueGt0 = 0.9
    beta_2: ConstrainedFloatValueGt0 = 0.999
    epsilon: ConstrainedFloatValueGt0 = 1e-07
