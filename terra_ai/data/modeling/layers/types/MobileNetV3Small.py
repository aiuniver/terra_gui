"""
## Тип слоя `MobileNetV3Small`
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
        "module": ModuleChoice.mobilenetv3small,
        "module_type": ModuleTypeChoice.keras_pretrained_model,
    }
)


class ParametersMainData(BaseMixinData):
    include_top: bool = True
    weights: Optional[PretrainedModelWeightsChoice] = 'imagenet'
    pooling: Optional[PretrainedModelPoolingChoice]
    trainable: bool = False


class ParametersExtraData(BaseMixinData):
    classes: PositiveInt = 1000
    classifier_activation: Optional[ActivationChoice] = ActivationChoice.softmax
    include_preprocessing: bool = False
    dropout_rate: float = 0.2
    minimalistic: bool = True  # почему-то не работает при minimalistic = False (причем в колабе работает)
    alpha: float = 1.0
    pass

