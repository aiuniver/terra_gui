"""
## Оптимайзер `Ftrl`
"""

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0, ConstrainedFloatValueLe0


class ParametersExtraData(BaseMixinData):
    learning_rate_power: ConstrainedFloatValueLe0 = -0.5
    initial_accumulator_value: ConstrainedFloatValueGe0 = 0.1
    l1_regularization_strength: ConstrainedFloatValueGe0 = 0.0
    l2_regularization_strength: ConstrainedFloatValueGe0 = 0.0
    l2_shrinkage_regularization_strength: ConstrainedFloatValueGe0 = 0.0
    beta: ConstrainedFloatValueGe0 = 0.0
