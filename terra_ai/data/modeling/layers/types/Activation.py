"""
## Тип слоя `Activation`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice


# LayerConfig = LayerConfigData(
#     num_uplinks=1,
#     num_uplinks_mode=DimModeTypeChoice.fixed,
#     input_dimension=2,
#     input_dim_mode=DimModeTypeChoice.minimal,
#     module=ModuleChoice.tensorflow_keras_layers,
#     module_type=ModuleTypeChoice.keras,
# )


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice


class ParametersExtraData(BaseMixinData):
    pass
