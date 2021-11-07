"""
## Тип слоя `ResNet50V2`
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
        "module": ModuleChoice.resnet50v2,
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
    classifier_activation: Optional[ActivationChoice] = ActivationChoice.softmax
    pass


# "output_layer": {
#     "type": "str",
#     "default": "last",
#     "list": True,
#     "available": ["conv2_block3_out",
#                   "conv3_block4_out",
#                   "conv4_block6_out",
#                   "last"],
# },