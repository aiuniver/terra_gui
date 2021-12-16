"""
## Тип слоя `PretrainedYOLO`
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
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 4,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    num_classes: PositiveInt = 5
    version: YOLOModeChoice = YOLOModeChoice.YOLOv3
    use_weights: bool = True


class ParametersExtraData(BaseMixinData):
    pass
