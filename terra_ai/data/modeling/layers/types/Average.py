"""
## Тип слоя `Average`
"""
from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import ActivationChoice
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: ConstrainedIntValueGe2 = 2
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
