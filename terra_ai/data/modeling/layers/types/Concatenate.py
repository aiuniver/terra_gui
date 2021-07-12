"""
## Тип слоя `Concatenate`
"""
from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1


class LayerConfig(BaseMixinData):
    num_uplinks: ConstrainedIntValueGe2 = 2
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras