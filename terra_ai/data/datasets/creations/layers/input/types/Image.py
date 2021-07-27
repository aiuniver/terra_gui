from typing import Optional
from pydantic.types import PositiveInt

from ...extra import FileInfo
from ...image_augmentation import AugmentationData
from ......mixins import BaseMixinData
from .....extra import LayerNetChoice, LayerScalerChoice


class ParametersData(BaseMixinData):
    file_info: FileInfo
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    put: Optional[str]
    object_detection: Optional[bool] = False
    augmentation: Optional[AugmentationData]
