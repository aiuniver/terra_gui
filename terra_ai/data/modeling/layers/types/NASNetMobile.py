"""
## Тип слоя `NASNetMobile`
"""
from typing import Optional

from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.layers.extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, \
    PretrainedModelWeightsChoice, PretrainedModelPoolingChoice, ActivationChoice

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
        "module": ModuleChoice.nasnetmobile,
        "module_type": ModuleTypeChoice.keras_pretrained_model,
    }
)


class ParametersMainData(BaseMixinData):
    include_top: bool = False
    weights: Optional[PretrainedModelWeightsChoice]
    pooling: Optional[PretrainedModelPoolingChoice]
    trainable: bool = False


class ParametersExtraData(BaseMixinData):
    classes: PositiveInt = 1000
    pass

