"""
## Тип слоя `Conv1DTranspose`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import (
    PaddingChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt
    kernel_size: PositiveInt
    strides: PositiveInt = 1
    padding: PaddingChoice = PaddingChoice.valid
    activation: Optional[ActivationChoice]


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
