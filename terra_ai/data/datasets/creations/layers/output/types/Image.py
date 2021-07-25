from typing import Optional
from pydantic.types import DirectoryPath, PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerNetChoice, LayerScalerChoice


class ParametersData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
