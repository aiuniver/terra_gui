"""
## Тип слоя `Attention`
"""

from ..extra import ModuleChoice, ModuleTypeChoice, LayerConfigData
from ....mixins import BaseMixinData


LayerConfig = LayerConfigData(
    num_uplinks=[2, 3],
    input_dimension=2,
    module=ModuleChoice.tensorflow_keras_layers,
    module_type=ModuleTypeChoice.keras,
)


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False
