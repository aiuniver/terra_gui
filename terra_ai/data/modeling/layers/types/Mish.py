"""
## Тип слоя `Mish`
"""
from pydantic import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.terra_custom_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.terra_layer
