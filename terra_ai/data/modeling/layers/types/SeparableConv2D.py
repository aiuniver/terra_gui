"""
## Тип слоя `SeparableConv2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import (
    PaddingChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 4
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    filters: PositiveInt
    kernel_size: Tuple[PositiveInt, PositiveInt]
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    padding: PaddingChoice = PaddingChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: Tuple[PositiveInt, PositiveInt] = (1, 1)
    depth_multiplier: PositiveInt = 1
    use_bias: bool = True
    depthwise_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    pointwise_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    depthwise_regularizer: Optional[RegularizerChoice]
    pointwise_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    depthwise_constraint: Optional[ConstraintChoice]
    pointwise_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
