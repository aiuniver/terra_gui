"""
## Тип слоя `ELU`
"""

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0Le1 = 1
