"""
## Тип слоя `AveragePooling3D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import PaddingChoice, DataFormatChoice


class ParametersMainData(BaseMixinData):
    pool_size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)
    strides: Optional[Tuple[PositiveInt, PositiveInt, PositiveInt]]
    padding: PaddingChoice = PaddingChoice.same


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
