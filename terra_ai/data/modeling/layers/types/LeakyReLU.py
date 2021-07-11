"""
## Тип слоя `LeakyReLU`
"""

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0


class ParametersMainData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0 = 0.3


class ParametersExtraData(BaseMixinData):
    pass
