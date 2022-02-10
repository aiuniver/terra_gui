"""
## Тип слоя `PretrainedModel`
"""

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.layers.extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
)


LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.terra_pretrained_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    model_path: str = ""
    load_weights: bool = False
    froze_model: bool = False


class ParametersExtraData(BaseMixinData):
    pass
