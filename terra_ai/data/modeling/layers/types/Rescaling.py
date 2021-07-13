"""
## Тип слоя `Rescaling`
"""

from ....mixins import BaseMixinData


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
