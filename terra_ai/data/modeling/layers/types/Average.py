"""
## Тип слоя `Average`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: ConstrainedIntValueGe2 = 2
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    pass
