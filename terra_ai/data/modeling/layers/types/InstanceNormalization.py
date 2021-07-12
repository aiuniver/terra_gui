"""
## Тип слоя `BatchNormalization`
"""

from typing import Optional
from pydantic.types import PositiveFloat

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice


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


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'customLayers'
    module_type: str = 'terra_layer'
