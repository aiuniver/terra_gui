"""
## Тип слоя `GRU`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1
from ..extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    ModuleChoice,
    ModuleTypeChoice,
)


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 3
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


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
    unroll: bool = False
    time_major: bool = False
    reset_after: bool = True
