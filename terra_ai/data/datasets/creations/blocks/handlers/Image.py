from typing import Optional
from pydantic import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerImageFrameModeChoice,
)


class ParametersData(BaseMixinData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice
    scaler: LayerScalerImageChoice
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch

    # Внутренние параметры
    put: Optional[PositiveInt]
    object_detection: Optional[bool] = False
