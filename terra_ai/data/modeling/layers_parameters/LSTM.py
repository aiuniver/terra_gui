"""
## Тип слоя `LSTM`
"""

from typing import Optional
from pydantic.types import confloat, PositiveInt

from ...mixins import BaseMixinData
from .extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    units: PositiveInt
    return_sequences: bool = False
    return_state: bool = False


class ParametersExtraData(BaseMixinData):
    activation: ActivationChoice = ActivationChoice.tanh
    recurrent_activation: ActivationChoice = ActivationChoice.sigmoid
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    recurrent_initializer: InitializerChoice = InitializerChoice.orthogonal
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    unit_forget_bias: bool = True
    kernel_regularizer: Optional[RegularizerChoice]
    recurrent_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    recurrent_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
    dropout: confloat(ge=0, le=1) = 0
    recurrent_dropout: confloat(ge=0, le=1) = 0
    go_backwards: bool = False
    stateful: bool = False
    time_major: bool = False
    unroll: bool = False
