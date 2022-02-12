"""
## Тип слоя `PSPBlock1D`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice,
)
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe1

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 3,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_base_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    filters_base: PositiveInt = 16
    n_pooling_branches: ConstrainedIntValueGe1 = 3
    filters_coef: ConstrainedIntValueGe1 = 1
    n_conv_layers: ConstrainedIntValueGe1 = 1
    activation: Optional[ActivationChoice] = ActivationChoice.relu
    batch_norm_layer: bool = True
    dropout_layer: bool = True


class ParametersExtraData(BaseMixinData):
    kernel_size: PositiveInt = 5
    dropout_rate: ConstrainedFloatValueGe0Le1 = 0.1
    pass
