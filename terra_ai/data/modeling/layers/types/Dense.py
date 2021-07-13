"""
## Тип слоя `Dense`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
)
from ....types import ConstrainedIntValueGe2


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    units: PositiveInt
    activation: Optional[ActivationChoice] = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    use_bias: bool = True
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    bias_initializer: InitializerChoice = InitializerChoice.zeros
    kernel_regularizer: Optional[RegularizerChoice]
    bias_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    kernel_constraint: Optional[ConstraintChoice]
    bias_constraint: Optional[ConstraintChoice]
