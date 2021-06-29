"""
## Тип слоя `Flatten`
"""

from typing import Optional

from ...mixins import BaseMixinData
from .extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: Optional[DataFormatChoice]
