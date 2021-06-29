"""
## Тип слоя `AveragePooling3D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import PaddingChoice, DataFormatChoice


class ParametersMainData(BaseMixinData):
    pool_size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)
    strides: Optional[PositiveInt]
    padding: PaddingChoice = PaddingChoice.valid


class ParametersExtraData(BaseMixinData):
    data_format: Optional[DataFormatChoice]
