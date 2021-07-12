"""
## Тип слоя `RepeatVector`
"""

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    n: PositiveInt
