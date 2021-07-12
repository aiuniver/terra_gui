"""
## Тип слоя `LSTM`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0Le1
from .extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    units: PositiveInt = 32
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
    dropout: ConstrainedFloatValueGe0Le1 = 0
    recurrent_dropout: ConstrainedFloatValueGe0Le1 = 0
    go_backwards: bool = False
    stateful: bool = False
    time_major: bool = False
    unroll: bool = False


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 3
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
