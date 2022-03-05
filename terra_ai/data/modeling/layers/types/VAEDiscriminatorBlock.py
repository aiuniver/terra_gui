"""
## Тип слоя `VAEDiscriminatorBlock`
"""
from typing import Optional

from pydantic import PositiveInt

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    VAELatentRegularizerChoice,
)
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0

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
        "module": ModuleChoice.terra_gan_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    conv_filters: PositiveInt = 128
    dense_units: PositiveInt = 1024
    # batch_size: PositiveInt = 32


class ParametersExtraData(BaseMixinData):
    use_bias: bool = True
    leaky_relu_alpha: ConstrainedFloatValueGe0 = 0.3
    bn_momentum: ConstrainedFloatValueGe0 = 0.99
    pass
