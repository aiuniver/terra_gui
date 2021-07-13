"""
## Тип слоя `Average`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
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
    activation: ActivationChoice = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    pass
