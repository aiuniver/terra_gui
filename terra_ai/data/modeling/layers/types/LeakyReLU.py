"""
## Тип слоя `LeakyReLU`
"""

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0 = 0.3


class ParametersExtraData(BaseMixinData):
    pass
