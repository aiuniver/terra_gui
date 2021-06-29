"""
## Тип слоя `DepthwiseConv2D`
"""

from typing import Tuple, Optional
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import (
    PaddingChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    kernel_size: Tuple[PositiveInt, PositiveInt]
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    padding: PaddingChoice = PaddingChoice.valid
    activation: Optional[ActivationChoice]


class ParametersExtraData(BaseMixinData):
    depth_multiplier: PositiveInt = 1
    data_format: Optional[DataFormatChoice]
    dilation_rate: Tuple[PositiveInt, PositiveInt] = (1, 1)
    use_bias: bool = True
    depthwise_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    depthwise_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    depthwise_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
