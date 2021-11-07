"""
## Тип слоя `YOLOConvBlock`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt

from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, ActivationChoice, \
    YOLOModeChoice, YOLOActivationChoice, PaddingChoice
from ....mixins import BaseMixinData

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 4,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    mode: YOLOModeChoice = YOLOModeChoice.YOLOv3
    filters: PositiveInt = 32
    num_conv: PositiveInt = 1


class ParametersExtraData(BaseMixinData):
    use_bias: bool = False
    activation: YOLOActivationChoice = YOLOActivationChoice.LeakyReLU
    first_conv_kernel: Tuple[PositiveInt, PositiveInt] = (3, 3)
    first_conv_strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    first_conv_padding: PaddingChoice = PaddingChoice.same
    include_bn_activation: bool = True
    pass
