"""
## Тип слоя `CustomBlock`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": None,
        "module_type": ModuleTypeChoice.block_plan,
    }
)


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    pass
