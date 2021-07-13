"""
## Тип слоя `DepthwiseConv2D`
"""

from typing import Tuple, Optional
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
    kernel_size: Tuple[PositiveInt, PositiveInt]
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    padding: PaddingChoice = PaddingChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    depth_multiplier: PositiveInt = 1
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: Tuple[PositiveInt, PositiveInt] = (1, 1)
    use_bias: bool = True
    depthwise_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    depthwise_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    depthwise_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
