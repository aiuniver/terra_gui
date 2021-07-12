"""
## Тип слоя `PReLU`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    alpha_initializer: InitializerChoice = InitializerChoice.zeros
    alpha_regularizer: Optional[RegularizerChoice]
    alpha_constraint: Optional[ConstraintChoice]
    shared_axes: Optional[Tuple[PositiveInt, ...]] = None


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
