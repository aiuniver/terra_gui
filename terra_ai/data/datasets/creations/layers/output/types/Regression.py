from typing import Optional
from pydantic.types import PositiveInt

from ...extra import FileInfo
from .....extra import LayerScalerChoice
from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    file_info: FileInfo
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
