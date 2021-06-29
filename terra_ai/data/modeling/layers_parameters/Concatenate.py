"""
## Тип слоя `Concatenate`
"""

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
