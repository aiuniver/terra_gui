"""
## Тип слоя `Conv2DTranspose`
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
    filters: PositiveInt = 32
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    padding: PaddingChoice = PaddingChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    output_padding: Optional[Tuple[PositiveInt, PositiveInt]]
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: Tuple[PositiveInt, PositiveInt] = (1, 1)
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
