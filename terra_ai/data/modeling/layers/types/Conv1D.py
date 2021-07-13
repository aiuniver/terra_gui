"""
## Тип слоя `Conv1D`
"""

from typing import Optional
from pydantic.types import PositiveInt, List

from ....mixins import BaseMixinData
from ..extra import (
    LayerConfigData,
    PaddingAddCausalChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    ModuleChoice,
    ModuleTypeChoice, DimModeTypeChoice,
)


LayerConfig = LayerConfigData(
    num_uplinks=1,
    num_uplinks_mode=DimModeTypeChoice.fixed,
    input_dimension=3,
    input_dim_mode=DimModeTypeChoice.minimal,
    module=ModuleChoice.tensorflow_keras_layers,
    module_type=ModuleTypeChoice.keras,
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt
    kernel_size: PositiveInt
    strides: PositiveInt = 1
    padding: PaddingAddCausalChoice = PaddingAddCausalChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    dilation_rate: PositiveInt = 1
    groups: PositiveInt = 1
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
