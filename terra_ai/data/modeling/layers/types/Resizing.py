"""
## Тип слоя `Resizing`
"""

from pydantic import PositiveInt

from ....mixins import BaseMixinData
from ..extra import ResizingInterpolationChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, \
    ModuleTypeChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 4,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers_preprocessing,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    height: PositiveInt = 224
    width: PositiveInt = 224


class ParametersExtraData(BaseMixinData):
    interpolation: ResizingInterpolationChoice = ResizingInterpolationChoice.bilinear
