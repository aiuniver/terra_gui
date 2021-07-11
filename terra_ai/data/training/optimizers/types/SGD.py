"""
## Оптимайзер `SGD`
"""

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0


class ParametersExtraData(BaseMixinData):
    momentum: ConstrainedFloatValueGe0 = 0.0
    nesterov: bool = False
