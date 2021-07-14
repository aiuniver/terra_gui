"""
## Тип слоя `AveragePooling1D`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import PaddingChoice, DataFormatChoice


class ParametersMainData(BaseMixinData):
    pool_size: PositiveInt = 2
    strides: Optional[PositiveInt]
    padding: PaddingChoice = PaddingChoice.valid


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
