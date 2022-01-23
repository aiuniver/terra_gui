"""
## Тип слоя `UNETBlock2D`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt, PositiveFloat

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice, InitializerChoice, NormalizationChoice, RegularizerChoice,
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
    n_pooling_branches: PositiveInt = 3
    activation: Optional[ActivationChoice] = ActivationChoice.relu
    normalization: Optional[NormalizationChoice] = NormalizationChoice.batch
    dropout_layer: bool = True
    leaky_relu_layer: bool = False
    use_activation_layer: bool = False
    maxpooling: bool = True
    upsampling: bool = True


class ParametersExtraData(BaseMixinData):
    filters_coef: PositiveInt = 1
    n_conv_layers: PositiveInt = 1
    use_bias: bool = True
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    kernel_regularizer: Optional[RegularizerChoice]
    dropout_rate: ConstrainedFloatValueGe0Le1 = 0.1
    leaky_relu_alpha: PositiveFloat = 0.3
    pass
