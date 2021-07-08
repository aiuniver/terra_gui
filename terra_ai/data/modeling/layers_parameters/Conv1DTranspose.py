"""
## Тип слоя `Conv1DTranspose`
"""

from typing import Optional
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
    kernel_size: PositiveInt = 5
    strides: PositiveInt = 2
    padding: PaddingChoice = PaddingChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    output_padding: Optional[PositiveInt]
    data_format: Optional[DataFormatChoice]
    dilation_rate: PositiveInt = 1
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
