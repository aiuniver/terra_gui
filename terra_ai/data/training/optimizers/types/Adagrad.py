"""
## Оптимайзер `Adagrad`
"""

from pydantic.types import PositiveFloat

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0


class ParametersExtraData(BaseMixinData):
    initial_accumulator_value: ConstrainedFloatValueGe0 = 0.1
    epsilon: PositiveFloat = 1e-07
