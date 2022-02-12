"""
## Тип слоя `RGBNormalization`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt, PositiveFloat

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice, CONVBlockConfigChoice, InitializerChoice, NormalizationChoice, RegularizerChoice,
)
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1

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
        "module": ModuleChoice.terra_normalization_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    use_div2k_mean: bool = False
    denormalize: bool = False
    half_range_normalization: bool = False


class ParametersExtraData(BaseMixinData):
    pass
