"""
## Тип слоя `LeakyReLU`
"""

from pydantic.types import confloat

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    alpha: confloat(ge=0) = 0.3


class ParametersExtraData(BaseMixinData):
    pass
