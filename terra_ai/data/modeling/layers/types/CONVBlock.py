"""
## Тип слоя `CONVBlock`
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
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    n_conv_layers: PositiveInt = 1
    transpose: bool = False
    filters: PositiveInt = 16
    activation: Optional[ActivationChoice] = ActivationChoice.relu
    normalization: Optional[NormalizationChoice] = NormalizationChoice.batch
    leaky_relu_layer: bool = False
    use_activation_layer: bool = False
    dropout_layer: bool = True


class ParametersExtraData(BaseMixinData):
    use_bias: bool = True
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    padding: PaddingChoice = PaddingChoice.same
    strides: Tuple[PositiveInt, PositiveInt] = (1, 1)
    dilation: Tuple[PositiveInt, PositiveInt] = (1, 1)
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    kernel_regularizer: Optional[RegularizerChoice]
    dropout_rate: ConstrainedFloatValueGe0Le1 = 0.1
    layers_seq_config: CONVBlockConfigChoice = CONVBlockConfigChoice.conv_bn_lrelu_drop_conv_bn_lrelu_drop
    leaky_relu_alpha: PositiveFloat = 0.3
    bn_momentum: ConstrainedFloatValueGe0Le1 = 0.99
    pass
