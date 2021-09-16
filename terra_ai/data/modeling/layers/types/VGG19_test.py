"""
## Тип слоя `VGG19`
"""
from typing import Optional

from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.layers.extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, \
    PretrainedModelWeightsChoice, PretrainedModelPoolingChoice, ActivationChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1, # (1 ветка сверху)?
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 4,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.vgg16, # сперва добавить vgg19 в class ModuleChoice?
        "module_type": ModuleTypeChoice.keras_pretrained_model,
    }
)


class ParametersMainData(BaseMixinData):
    include_top: bool = False
    weights: Optional[PretrainedModelWeightsChoice] # выбирать не из чего?
    pooling: Optional[PretrainedModelPoolingChoice] # --//--
    trainable: bool = False


class ParametersExtraData(BaseMixinData):
    classes: PositiveInt = 1000
    classifier_activation: Optional[ActivationChoice] = ActivationChoice.softmax # здесь задаются параметры по умолчанию?
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