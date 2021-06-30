"""
## Оптимайзер `Adagrad`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0, ConstrainedFloatValueGt0


class ParametersExtraData(BaseMixinData):
    initial_accumulator_value: ConstrainedFloatValueGe0 = 0.1
    epsilon: ConstrainedFloatValueGt0 = 1e-07
