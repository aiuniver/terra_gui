"""
## Тип слоя `ConvLSTM2D`
"""

from typing import Tuple, Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import (
    PaddingChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
)
from ....types import ConstrainedFloatValueGe0Le1

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 5,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt = 32
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    padding: PaddingChoice = PaddingChoice.same
    activation: ActivationChoice = ActivationChoice.tanh
    recurrent_activation: ActivationChoice = ActivationChoice.hard_sigmoid


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: Tuple[PositiveInt, PositiveInt] = (1, 1)
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
    return_sequences: bool = False
    return_state: bool = False
    go_backwards: bool = False
    stateful: bool = False
    dropout: ConstrainedFloatValueGe0Le1 = 0.0
    recurrent_dropout: ConstrainedFloatValueGe0Le1 = 0.0
