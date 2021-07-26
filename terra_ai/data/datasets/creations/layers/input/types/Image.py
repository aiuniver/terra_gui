from typing import Optional
from pydantic.types import DirectoryPath, PositiveInt

from ...extra import FileInfo
from ......mixins import BaseMixinData
from .....extra import LayerNetChoice, LayerScalerChoice


class ParametersData(BaseMixinData):
    file_info: FileInfo
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
