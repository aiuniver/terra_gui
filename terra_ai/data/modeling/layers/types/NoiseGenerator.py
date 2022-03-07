"""
## Тип слоя `NoiseGenerator`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt, PositiveFloat

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice, CONVBlockConfigChoice, InitializerChoice, NormalizationChoice, RegularizerChoice,
    NoiseTypeChoice,
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
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_normalization_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    noise_level: ConstrainedFloatValueGe0Le1 = 0.5
    noise_type: NoiseTypeChoice = NoiseTypeChoice.uniform


class ParametersExtraData(BaseMixinData):
    mean: float = 0.
    stddev: float = 1.
    minval: float = 0.
    maxval: float = 1.
