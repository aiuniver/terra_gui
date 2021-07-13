"""
## Тип слоя `ZeroPadding2D`
"""

from typing import Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

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
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    padding: Tuple[Tuple[PositiveInt, PositiveInt], Tuple[PositiveInt, PositiveInt]]


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
