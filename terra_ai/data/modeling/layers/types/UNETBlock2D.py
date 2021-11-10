"""
## Тип слоя `UNETBlock`
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
from ....types import ConstrainedFloatValueGe0Le1

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
    filters_base: PositiveInt = 16
    n_pooling_branches: PositiveInt = 2
    filters_coef: PositiveInt = 1
    n_conv_layers: PositiveInt = 2
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    padding: PaddingChoice = PaddingChoice.same
    activation: Optional[ActivationChoice] = ActivationChoice.relu
    batch_norm_layer: bool = True
    dropout_layer: bool = True


class ParametersExtraData(BaseMixinData):
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    dilation: Tuple[PositiveInt, PositiveInt] = (1, 1)
    dropout_rate: ConstrainedFloatValueGe0Le1 = 0.1
    pass
