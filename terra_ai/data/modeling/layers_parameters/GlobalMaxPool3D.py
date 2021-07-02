"""
## Тип слоя `GlobalMaxPool3D`
"""

from typing import Optional

from ...mixins import BaseMixinData
from .extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
