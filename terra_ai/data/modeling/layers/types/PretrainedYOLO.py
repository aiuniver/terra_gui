"""
## Тип слоя `PretrainedYOLO`
"""

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, YOLOModeChoice

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
    mode: YOLOModeChoice = YOLOModeChoice.YOLOv3


class ParametersExtraData(BaseMixinData):
    pass
