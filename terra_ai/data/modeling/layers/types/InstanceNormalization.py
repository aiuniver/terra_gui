"""
## Тип слоя `BatchNormalization`
"""

from typing import Optional
from pydantic.types import PositiveFloat, PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe2
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.terra_custom_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.terra_layer


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
    epsilon: PositiveFloat = 0.001
    center: bool = True
    scale: bool = True
    beta_initializer: InitializerChoice = InitializerChoice.zeros
    gamma_initializer: InitializerChoice = InitializerChoice.ones
    beta_regularizer: Optional[RegularizerChoice]
    gamma_regularizer: Optional[RegularizerChoice]
    beta_constraint: Optional[ConstraintChoice]
    gamma_constraint: Optional[ConstraintChoice]
