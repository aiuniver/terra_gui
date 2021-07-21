"""
## Тип слоя `BatchNormalization`
"""

from typing import Optional
from pydantic.types import PositiveFloat

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice, LayerConfigData, \
    LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

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
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
    momentum: ConstrainedFloatValueGe0Le1 = 0.99
    epsilon: PositiveFloat = 0.001
    center: bool = True
    scale: bool = True
    beta_initializer: InitializerChoice = InitializerChoice.zeros
    gamma_initializer: InitializerChoice = InitializerChoice.ones
    moving_mean_initializer: InitializerChoice = InitializerChoice.zeros
    moving_variance_initializer: InitializerChoice = InitializerChoice.ones
    beta_regularizer: Optional[RegularizerChoice]
    gamma_regularizer: Optional[RegularizerChoice]
    beta_constraint: Optional[ConstraintChoice]
    gamma_constraint: Optional[ConstraintChoice]
