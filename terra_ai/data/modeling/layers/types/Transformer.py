"""
## Тип слоя `Transformer`
"""

from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.layers.extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    YOLOModeChoice,
)

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 2,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
        embed_dim: PositiveInt = 256
        latent_dim: PositiveInt = 2048
        num_heads: PositiveInt = 8
        vocab_size: PositiveInt = 15000

class ParametersExtraData(BaseMixinData):
    pass
