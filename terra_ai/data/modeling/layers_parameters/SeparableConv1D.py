"""
## Тип слоя `SeparableConv1D`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import (
    PaddingAddCausalChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt = 32
    kernel_size: PositiveInt = 3
    strides: PositiveInt = 1
    padding: PaddingAddCausalChoice = PaddingAddCausalChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: PositiveInt = 1
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
