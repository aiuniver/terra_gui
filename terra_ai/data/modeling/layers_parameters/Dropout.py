"""
## Тип слоя `Dropout`
"""

from typing import Optional
from pydantic.types import confloat, PositiveInt

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    rate: confloat(ge=0, le=1)


class ParametersExtraData(BaseMixinData):
    noise_shape: Optional[PositiveInt]
    seed: Optional[PositiveInt]
