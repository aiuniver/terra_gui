"""
## Тип слоя `Cropping2D`
"""

from typing import Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 4
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    cropping: Tuple[Tuple[PositiveInt, PositiveInt], Tuple[PositiveInt, PositiveInt]]


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
