"""
## Тип слоя `VGG16`
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
        "module": ModuleChoice.vgg16,
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
    pass


#
# "output_layer": {
#     "type": "str",
#     "default": "last",
#     "list": True,
#     "available": ["block1_conv2",
#                   "block2_conv2",
#                   "block3_conv3",
#                   "block4_conv3",
#                   "block5_conv3",
#                   "last"],
# },