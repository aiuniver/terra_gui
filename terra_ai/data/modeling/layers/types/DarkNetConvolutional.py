 """
## Тип слоя `DarkNetConvolutional`
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
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    filters: PositiveInt = 32
    activate_type: YOLOActivationChoice = YOLOActivationChoice.LeakyReLU
    pass


class ParametersExtraData(BaseMixinData):
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    downsample: bool = False
    activate: bool = True
    bn: bool = True
    pass
