"""
## Тип слоя `UpSampling2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import DataFormatChoice, InterpolationChoice


class ParametersMainData(BaseMixinData):
    size: Tuple[PositiveInt, PositiveInt] = (2, 2)


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    interpolation: InterpolationChoice = InterpolationChoice.nearest
