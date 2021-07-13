"""
## Тип слоя `Activation`
"""
from pydantic.types import PositiveInt, ConstrainedInt

from ..extra import ModuleChoice, ModuleTypeChoice, LayerConfigData, DimModeTypeChoice
from ....mixins import BaseMixinData
from ..extra import ActivationChoice
from ....types import ConstrainedIntValueGe2


LayerConfig = LayerConfigData(
    num_uplinks=1,
    num_uplinks_mode=DimModeTypeChoice.fixed,
    input_dimension=2,
    input_dim_mode=DimModeTypeChoice.minimal,
    module=ModuleChoice.tensorflow_keras_layers,
    module_type=ModuleTypeChoice.keras,
)

class ParametersMainData(BaseMixinData):
    activation: ActivationChoice


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
