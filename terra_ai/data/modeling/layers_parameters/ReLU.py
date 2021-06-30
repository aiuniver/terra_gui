"""
## Тип слоя `ReLU`
"""

from typing import Optional

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    max_value: Optional[ConstrainedFloatValueGe0]
    negative_slope: ConstrainedFloatValueGe0 = 0
    threshold: ConstrainedFloatValueGe0 = 0
