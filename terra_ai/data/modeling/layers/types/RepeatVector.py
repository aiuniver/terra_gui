"""
## Тип слоя `RepeatVector`
"""

from pydantic.types import PositiveInt

from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    n: PositiveInt
