"""
## Тип слоя `Activation`
"""

from ...mixins import BaseMixinData
from .extra import ActivationChoice


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    pass
