"""
## Тип слоя `Normalization`
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
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers_preprocessing
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
    mean: Optional[float] = None
    variance: Optional[float] = None
