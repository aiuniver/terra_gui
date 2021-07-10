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
