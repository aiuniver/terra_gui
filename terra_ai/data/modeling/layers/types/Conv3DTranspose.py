"""
## Тип слоя `Conv3DTranspose`
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
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt
    kernel_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    strides: Tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    padding: PaddingChoice = PaddingChoice.valid
    activation: Optional[ActivationChoice]


class ParametersExtraData(BaseMixinData):
    output_padding: Optional[Tuple[PositiveInt, PositiveInt, PositiveInt]]
    data_format: Optional[DataFormatChoice]
    dilation_rate: Tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
