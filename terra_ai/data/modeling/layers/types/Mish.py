"""
## Тип слоя `Mish`
"""

from ....mixins import BaseMixinData


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.terra_custom_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.terra_layer


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    pass
