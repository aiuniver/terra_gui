"""
## Тип слоя `ResnetBlock2D`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    PaddingChoice, ActivationChoice, InitializerChoice, ResblockActivationChoice, NormalizationChoice,
    MergeLayerChoice, RegularizerChoice,
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
    filters: PositiveInt = 32
    num_resblocks: PositiveInt = 1
    n_conv_layers: PositiveInt = 2
    use_activation_layer: bool = True
    activation: ResblockActivationChoice = ResblockActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    kernel_size: Tuple[PositiveInt, PositiveInt] = (3, 3)
    kernel_initializer: InitializerChoice = InitializerChoice.glorot_uniform
    kernel_regularizer: Optional[RegularizerChoice]
    normalization: Optional[NormalizationChoice] = NormalizationChoice.batch
    merge_layer: MergeLayerChoice = MergeLayerChoice.concatenate
    use_bias: bool = True
    bn_momentum: ConstrainedFloatValueGe0Le1 = 0.99
    prelu_shared_axes: Optional[Tuple[PositiveInt, ...]] = None
    pass
