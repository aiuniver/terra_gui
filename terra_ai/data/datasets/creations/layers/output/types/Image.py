from typing import Optional, List
from pydantic.types import PositiveInt

from ...image_augmentation import AugmentationData
from ...extra import ParametersBaseData
from .....extra import LayerNetChoice, LayerScalerChoice


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    put: Optional[str]
    object_detection: Optional[bool] = False
    augmentation: Optional[AugmentationData]
