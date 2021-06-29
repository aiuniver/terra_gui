"""
## Тип слоя `ELU`
"""

from pydantic.types import confloat

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    alpha: confloat(ge=0, le=1) = 1
