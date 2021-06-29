"""
## Тип слоя `Dense`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    units: PositiveInt
    activation: Optional[ActivationChoice]


class ParametersExtraData(BaseMixinData):
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
