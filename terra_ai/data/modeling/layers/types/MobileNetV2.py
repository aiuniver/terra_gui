"""
## Тип слоя `MobileNetV2`
"""
from typing import Optional

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, \
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
        "module": ModuleChoice.mobilenetv2,
        "module_type": ModuleTypeChoice.keras_pretrained_model,
    }
)


class ParametersMainData(BaseMixinData):
    include_top: bool = True
    weights: Optional[PretrainedModelWeightsChoice]
    pooling: Optional[PretrainedModelPoolingChoice]
    trainable: bool = False


class ParametersExtraData(BaseMixinData):
    classes: PositiveInt = 1000
    classifier_activation: Optional[ActivationChoice] = ActivationChoice.softmax
    alpha: float = 1.0
    pass

