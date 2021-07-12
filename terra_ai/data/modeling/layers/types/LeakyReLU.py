"""
## Тип слоя `LeakyReLU`
"""
from pydantic import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0, ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0 = 0.3


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras