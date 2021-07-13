"""
## Тип слоя `Attention`
"""
from ..extra import ModuleChoice, ModuleTypeChoice, LayerConfigData, DimModeTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe2

LayerConfig = LayerConfigData(
    num_uplinks=[2, 3],
    num_uplinks_mode=DimModeTypeChoice.fixed,
    input_dimension=2,
    input_dim_mode=DimModeTypeChoice.minimal,
    module=ModuleChoice.tensorflow_keras_layers,
    module_type=ModuleTypeChoice.keras,
)

class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False


# class LayerConfig(BaseMixinData):
#     num_uplinks: list = [2, 3]
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras