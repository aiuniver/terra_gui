"""
## Тип слоя `Dense`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import (
    ActivationChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice,
)

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


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
