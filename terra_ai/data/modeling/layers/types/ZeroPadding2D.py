"""
## Тип слоя `ZeroPadding2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import DataFormatChoice, InterpolationChoice


class ParametersMainData(BaseMixinData):
    padding: Tuple[Tuple[PositiveInt, PositiveInt], Tuple[PositiveInt, PositiveInt]]


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 4
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras