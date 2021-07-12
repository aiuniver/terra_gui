"""
## Тип слоя `Cropping2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import DataFormatChoice, InterpolationChoice


class ParametersMainData(BaseMixinData):
    cropping: Tuple[Tuple[PositiveInt, PositiveInt], Tuple[PositiveInt, PositiveInt]] = ((0, 0), (0, 0))


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 4
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
