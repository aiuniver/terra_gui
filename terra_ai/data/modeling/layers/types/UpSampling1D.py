"""
## Тип слоя `UpSampling1D`
"""

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    size: PositiveInt = 2


class ParametersExtraData(BaseMixinData):
    pass
