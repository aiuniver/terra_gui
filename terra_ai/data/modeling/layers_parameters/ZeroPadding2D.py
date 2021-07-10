"""
## Тип слоя `ZeroPadding2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import DataFormatChoice, InterpolationChoice


class ParametersMainData(BaseMixinData):
    padding: Tuple[Tuple[PositiveInt, PositiveInt], Tuple[PositiveInt, PositiveInt]] = ((1, 1), (1, 1))


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
