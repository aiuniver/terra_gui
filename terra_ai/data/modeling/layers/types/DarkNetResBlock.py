"""
## Тип слоя `DarkNetResBlock`
"""
from typing import Tuple

from pydantic import PositiveInt

from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, YOLOActivationChoice
from ....mixins import BaseMixinData

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
        "module": ModuleChoice.terra_yolo_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    filter_num1: PositiveInt = 32
    filter_num2: PositiveInt = 32
    activate_type: YOLOActivationChoice = YOLOActivationChoice.LeakyReLU
    pass


class ParametersExtraData(BaseMixinData):
    pass
