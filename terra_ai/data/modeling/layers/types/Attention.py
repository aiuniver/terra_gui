"""
## Тип слоя `Attention`
"""
from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": (2, 3),
            "validation": LayerValidationMethodChoice.dependence_tuple2,
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
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False
