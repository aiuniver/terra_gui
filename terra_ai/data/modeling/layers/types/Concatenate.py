"""
## Тип слоя `Concatenate`
"""

from ....mixins import BaseMixinData


# class LayerConfig(BaseMixinData):
#     num_uplinks: ConstrainedIntValueGe2 = 2
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1
