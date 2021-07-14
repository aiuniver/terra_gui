"""
## Тип слоя `Dropout`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1


class ParametersMainData(BaseMixinData):
    rate: ConstrainedFloatValueGe0Le1


class ParametersExtraData(BaseMixinData):
    noise_shape: Optional[PositiveInt]
    seed: Optional[PositiveInt]
