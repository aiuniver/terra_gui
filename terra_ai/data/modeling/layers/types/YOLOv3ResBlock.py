"""
## Тип слоя `YOLOv3ResBlock`
"""
from typing import Optional

from pydantic import PositiveInt

from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, ActivationChoice, \
    YOLOModeChoice, YOLOActivationChoice
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
    num_resblocks: PositiveInt = 1


class ParametersExtraData(BaseMixinData):
    use_bias: bool = False
    include_head: bool = True
    pass
