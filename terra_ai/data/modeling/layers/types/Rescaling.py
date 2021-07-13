"""
## Тип слоя `Rescaling`
"""

from typing import Optional
from pydantic.types import PositiveFloat, PositiveInt

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe2
from ..extra import (
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    ModuleChoice,
    ModuleTypeChoice,
)


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers_preprocessing
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    scale: float = 1.0
    offset: float = 0.0


class ParametersExtraData(BaseMixinData):
    pass
