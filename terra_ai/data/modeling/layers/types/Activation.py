"""
## Тип слоя `Activation`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice


class ParametersExtraData(BaseMixinData):
    pass
