"""
## Тип слоя `BatchNormalization`
"""

from typing import Optional

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0Le1, ConstrainedFloatValueGt0
from .extra import InitializerChoice, RegularizerChoice, ConstraintChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
    momentum: ConstrainedFloatValueGe0Le1 = 0.99
    epsilon: ConstrainedFloatValueGt0 = 0.001
    center: bool = True
    scale: bool = True
    beta_initializer: InitializerChoice = InitializerChoice.zeros
    gamma_initializer: InitializerChoice = InitializerChoice.ones
    moving_mean_initializer: InitializerChoice = InitializerChoice.zeros
    moving_variance_initializer: InitializerChoice = InitializerChoice.ones
    beta_regularizer: Optional[RegularizerChoice]
    gamma_regularizer: Optional[RegularizerChoice]
    beta_constraint: Optional[ConstraintChoice]
    gamma_constraint: Optional[ConstraintChoice]
