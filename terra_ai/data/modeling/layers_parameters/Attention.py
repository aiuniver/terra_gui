"""
## Тип слоя `Attention`
"""

from ...mixins import BaseMixinData
from .extra import ActivationChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False
