"""
## Тип слоя `ConditionalMergeLayer`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt, PositiveFloat

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice, CONVBlockConfigChoice, ConditionalMergeModeChoice,
)
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 2,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    mode: ConditionalMergeModeChoice = ConditionalMergeModeChoice.concatenate


class ParametersExtraData(BaseMixinData):
    pass
