"""
## Тип слоя `InceptionV3`
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
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.inception_v3,
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
#     "available": ["mixed0",
#                   "mixed1",
#                   "mixed2",
#                   "mixed3",
#                   "mixed4",
#                   "mixed5",
#                   "mixed6",
#                   "mixed7",
#                   "mixed8",
#                   "mixed9",
#                   "last"],
# },
